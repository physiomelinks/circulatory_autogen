// bdf_loop.cpp — Semi-implicit BDF loop using kernel replay with caching
#include "pyaadc.hpp"
#include <vector>
#include <map>
#include <tuple>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <aadc/serialization.h>

namespace py = pybind11;

class KernelExtFunc : public aadc::ConstStateExtFunc {
public:
    KernelExtFunc(
        std::shared_ptr<Functions> kernel,
        const std::vector<Argument>& kx, const std::vector<Argument>& kp,
        const std::vector<Result>& kr,
        const std::vector<aadc::ExtVarIndex>& tx, const std::vector<aadc::ExtVarIndex>& tp,
        const std::vector<aadc::ExtVarIndex>& to,
        std::shared_ptr<WorkSpace> shared_ws = nullptr
    ) : kernel_(kernel), kx_(kx), kp_(kp), kr_(kr), tx_(tx), tp_(tp), to_(to) {
        ws_ = shared_ws ? shared_ws : kernel_->createWorkSpace();
    }
    template<typename M> void forward(M* v) const {
        for (int a = 0; a < aadc::mmSize<M>(); a++) {
            for (size_t i=0;i<kx_.size();i++) aadc::toDblPtr(ws_->val(kx_[i]))[0]=aadc::toDblPtr(v[tx_[i]])[a];
            for (size_t k=0;k<kp_.size();k++) aadc::toDblPtr(ws_->val(kp_[k]))[0]=aadc::toDblPtr(v[tp_[k]])[a];
            kernel_->forward(*ws_);
            for (size_t i=0;i<kr_.size();i++) aadc::toDblPtr(v[to_[i]])[a]=aadc::toDblPtr(ws_->val(kr_[i]))[0];
        }
    }
    template<class M> void reverse(const M* v, M* d) const {
        for (int a = 0; a < aadc::mmSize<M>(); a++) {
            for (size_t i=0;i<kx_.size();i++) aadc::toDblPtr(ws_->val(kx_[i]))[0]=aadc::toDblPtr(v[tx_[i]])[a];
            for (size_t k=0;k<kp_.size();k++) aadc::toDblPtr(ws_->val(kp_[k]))[0]=aadc::toDblPtr(v[tp_[k]])[a];
            kernel_->forward(*ws_);
            ws_->resetDiff();
            for (size_t i=0;i<kr_.size();i++) aadc::toDblPtr(ws_->diff(kr_[i]))[0]=aadc::toDblPtr(d[to_[i]])[a];
            kernel_->reverse(*ws_);
            for (size_t i=0;i<kx_.size();i++) aadc::toDblPtr(d[tx_[i]])[a]+=aadc::toDblPtr(ws_->diff(kx_[i]))[0];
            for (size_t k=0;k<kp_.size();k++) aadc::toDblPtr(d[tp_[k]])[a]+=aadc::toDblPtr(ws_->diff(kp_[k]))[0];
        }
    }
private:
    std::shared_ptr<Functions> kernel_;
    std::vector<Argument> kx_,kp_; std::vector<Result> kr_;
    std::vector<aadc::ExtVarIndex> tx_,tp_,to_;
    mutable std::shared_ptr<WorkSpace> ws_;
};

// LU solve as ConstStateExtFunc: forward = LU\rhs, reverse = LU^{-T} adjoint
// This avoids recording ~750 idouble ops per Newton step on tape.
class LUSolveExtFunc : public aadc::ConstStateExtFunc {
public:
    LUSolveExtFunc(
        int n,
        const std::vector<std::vector<double>>& LU,
        const std::vector<int>& piv,
        const std::vector<aadc::ExtVarIndex>& rhs_idx,  // inputs (F)
        const std::vector<aadc::ExtVarIndex>& out_idx    // outputs (dy)
    ) : n_(n), LU_(LU), piv_(piv), rhs_idx_(rhs_idx), out_idx_(out_idx) {}

    template<typename M> void forward(M* v) const {
        for (int a = 0; a < aadc::mmSize<M>(); a++) {
            // Read rhs
            std::vector<double> b(n_);
            for (int i=0;i<n_;i++) b[i] = aadc::toDblPtr(v[rhs_idx_[i]])[a];
            // Permute
            std::vector<double> pb(n_);
            for (int i=0;i<n_;i++) pb[i] = b[piv_[i]];
            // Forward sub (L)
            for (int i=1;i<n_;i++)
                for (int j=0;j<i;j++) pb[i] -= LU_[i][j]*pb[j];
            // Back sub (U)
            for (int i=n_-1;i>=0;i--) {
                for (int j=i+1;j<n_;j++) pb[i] -= LU_[i][j]*pb[j];
                pb[i] /= LU_[i][i];
            }
            // Write output
            for (int i=0;i<n_;i++) aadc::toDblPtr(v[out_idx_[i]])[a] = pb[i];
        }
    }

    template<class M> void reverse(const M* v, M* d) const {
        for (int a = 0; a < aadc::mmSize<M>(); a++) {
            // Read output adjoint d_out
            std::vector<double> dout(n_);
            for (int i=0;i<n_;i++) dout[i] = aadc::toDblPtr(d[out_idx_[i]])[a];
            // Reverse of LU solve = (LU)^{-T} * dout
            // = U^{-T} L^{-T} P dout? No: reverse of Ax=b is A^T lambda = dout, drhs += P^T lambda
            // Actually: forward is x = (LU)^{-1} P b
            // dx/db = (LU)^{-1} P
            // d_rhs += P^T (LU)^{-T} d_out
            // Step 1: solve U^T z = d_out
            std::vector<double> z(n_);
            for (int i=0;i<n_;i++) z[i] = dout[i];
            for (int i=0;i<n_;i++) {
                z[i] /= LU_[i][i];
                for (int j=i+1;j<n_;j++) z[j] -= LU_[i][j]*z[i];
            }
            // Step 2: solve L^T w = z
            for (int i=n_-2;i>=0;i--)
                for (int j=i+1;j<n_;j++) z[i] -= LU_[j][i]*z[j];
            // Step 3: apply P^T (un-permute)
            // piv maps row i -> original row piv[i]
            // P^T maps: d_rhs[piv[i]] += z[i]
            for (int i=0;i<n_;i++)
                aadc::toDblPtr(d[rhs_idx_[piv_[i]]])[a] += z[i];
        }
    }

private:
    int n_;
    std::vector<std::vector<double>> LU_;
    std::vector<int> piv_;
    std::vector<aadc::ExtVarIndex> rhs_idx_, out_idx_;
};

// Step kernel ExtFunc: wraps a step kernel (compute_rates + BDF step) with constant t and J_diag.
// Gradient flows through x and p only (t and J_diag are off-tape constants).
class StepExtFunc : public aadc::ConstStateExtFunc {
public:
    std::shared_ptr<Functions> kern;
    std::vector<Argument> kx,kp; Argument kt; std::vector<Argument> kj;
    std::vector<Result> kr;
    std::vector<aadc::ExtVarIndex> tx,tp,to;
    double t_val; std::vector<double> j_val;
    mutable std::shared_ptr<WorkSpace> ws;
    int n_, np_;

    StepExtFunc(std::shared_ptr<Functions> k,
        const std::vector<Argument>& kx_, const std::vector<Argument>& kp_,
        Argument kt_, const std::vector<Argument>& kj_, const std::vector<Result>& kr_,
        const std::vector<aadc::ExtVarIndex>& tx_, const std::vector<aadc::ExtVarIndex>& tp_,
        const std::vector<aadc::ExtVarIndex>& to_,
        double tv, const std::vector<double>& jv, std::shared_ptr<WorkSpace> w)
        : kern(k), kx(kx_), kp(kp_), kt(kt_), kj(kj_), kr(kr_),
          tx(tx_), tp(tp_), to(to_), t_val(tv), j_val(jv), ws(w),
          n_(kx_.size()), np_(kp_.size()) {}

    template<typename M> void forward(M* v) const {
        for (int a=0;a<aadc::mmSize<M>();a++) {
            for (int i=0;i<n_;i++) aadc::toDblPtr(ws->val(kx[i]))[0]=aadc::toDblPtr(v[tx[i]])[a];
            for (int k=0;k<np_;k++) aadc::toDblPtr(ws->val(kp[k]))[0]=aadc::toDblPtr(v[tp[k]])[a];
            aadc::toDblPtr(ws->val(kt))[0]=t_val;
            for (int i=0;i<n_;i++) aadc::toDblPtr(ws->val(kj[i]))[0]=j_val[i];
            kern->forward(*ws);
            for (int i=0;i<n_;i++) aadc::toDblPtr(v[to[i]])[a]=aadc::toDblPtr(ws->val(kr[i]))[0];
        }
    }
    template<class M> void reverse(const M* v, M* d) const {
        for (int a=0;a<aadc::mmSize<M>();a++) {
            for (int i=0;i<n_;i++) aadc::toDblPtr(ws->val(kx[i]))[0]=aadc::toDblPtr(v[tx[i]])[a];
            for (int k=0;k<np_;k++) aadc::toDblPtr(ws->val(kp[k]))[0]=aadc::toDblPtr(v[tp[k]])[a];
            aadc::toDblPtr(ws->val(kt))[0]=t_val;
            for (int i=0;i<n_;i++) aadc::toDblPtr(ws->val(kj[i]))[0]=j_val[i];
            kern->forward(*ws);
            ws->resetDiff();
            for (int i=0;i<n_;i++) aadc::toDblPtr(ws->diff(kr[i]))[0]=aadc::toDblPtr(d[to[i]])[a];
            kern->reverse(*ws);
            for (int i=0;i<n_;i++) aadc::toDblPtr(d[tx[i]])[a]+=aadc::toDblPtr(ws->diff(kx[i]))[0];
            for (int k=0;k<np_;k++) aadc::toDblPtr(d[tp[k]])[a]+=aadc::toDblPtr(ws->diff(kp[k]))[0];
            // t and J_diag: no gradient (constants for AD)
        }
    }
};

// Static cache
static struct {
    std::shared_ptr<Functions> funcs;
    std::shared_ptr<WorkSpace> ws;
    std::vector<Argument> x_args, p_args;
    Result cost_res;
    int total_subs = 0;
    int newton_mode = -1;
} cache;

bool bdf_save_tape(const std::string& path) {
    if (!cache.funcs) return false;
    try {
        std::ofstream ofs(path, std::ios::binary);
        boost::archive::binary_oarchive ar(ofs);
        serialize_AADCFunctions(ar, *cache.funcs, 0);
        ar & cache.x_args & cache.p_args & cache.cost_res;
        ar & cache.total_subs & cache.newton_mode;
        return true;
    } catch (...) { return false; }
}

bool bdf_load_tape(const std::string& path) {
    try {
        auto funcs = std::make_shared<Functions>();
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return false;
        boost::archive::binary_iarchive ar(ifs);
        serialize_AADCFunctions(ar, *funcs, 0);
        std::vector<Argument> xa, pa;
        Result cr;
        int ts, nm;
        ar & xa & pa & cr & ts & nm;
        cache.funcs = funcs;
        cache.ws = funcs->createWorkSpace();
        cache.x_args = xa; cache.p_args = pa; cache.cost_res = cr;
        cache.total_subs = ts; cache.newton_mode = nm;
        return true;
    } catch (...) { return false; }
}

py::tuple bdf_record_and_evaluate(
    py::function compute_rates_fn, py::list py_states, py::list py_variables,
    py::list py_param_indices, py::list py_param_values,
    int total_steps, int pre_steps, int n_sub, double idt,
    py::list py_obs_list, py::object compute_variables_fn, int jac_lag,
    int newton_iters, py::list py_jac_coloring
) {
    int n = py::len(py_states), n_p = py::len(py_param_indices), n_vars = py::len(py_variables);
    std::vector<int> pidx(n_p);
    for (int k=0;k<n_p;k++) pidx[k] = py_param_indices[k].cast<int>();

    // Parse coloring: list of lists of column indices
    std::vector<std::vector<int>> jac_coloring;
    for (auto group : py_jac_coloring)
        jac_coloring.push_back(group.cast<std::vector<int>>());
    int total_subs = total_steps * n_sub;

    if (!cache.funcs || cache.total_subs != total_subs || cache.newton_mode != newton_iters) {
        // ---- Record f-kernel (compute_rates only, for Jacobian computation) ----
        auto kernel = std::make_shared<Functions>();
        kernel->startRecording();
        std::vector<idouble> kx(n); std::vector<Argument> kxa(n);
        for (int i=0;i<n;i++) { kx[i]=py_states[i].cast<double>(); kxa[i]=kx[i].markAsInput(); }
        std::vector<idouble> kv(n_vars); std::vector<Argument> kpa(n_p);
        for (int i=0;i<n_vars;i++) { double v=py_variables[i].cast<double>(); kv[i]=(v==v)?v:0.0; }
        for (int k=0;k<n_p;k++) { kv[pidx[k]]=py_param_values[k].cast<double>(); kpa[k]=kv[pidx[k]].markAsInput(); }
        idouble kt(0.0); Argument kta = kt.markAsInput();
        py::list pkx(n),pkr(n),pkv(n_vars);
        for (int i=0;i<n;i++) pkx[i]=py::cast(kx[i]);
        for (int i=0;i<n;i++) pkr[i]=py::cast(idouble(0.0));
        for (int i=0;i<n_vars;i++) pkv[i]=py::cast(kv[i]);
        compute_rates_fn(py::cast(kt),pkx,pkr,pkv);
        std::vector<Result> krr(n);
        for (int i=0;i<n;i++) {
            auto o=pkr[i]; idouble rv=py::isinstance<idouble>(o)?o.cast<idouble>():idouble(o.cast<double>());
            krr[i]=rv.markAsOutput();
        }
        kernel->stopRecording();

        // ---- Record step kernel (compute_rates + semi-implicit, ONE step) ----
        // Inputs: x(n), p(n_p), t(1), J_diag(n). Outputs: x_new(n).
        // This kernel is replayed 22000 times — NO idouble arithmetic on main tape.
        auto step_kernel = std::make_shared<Functions>();
        step_kernel->startRecording();
        std::vector<idouble> sx(n); std::vector<Argument> sxa(n);
        for (int i=0;i<n;i++) { sx[i]=py_states[i].cast<double>(); sxa[i]=sx[i].markAsInput(); }
        std::vector<idouble> sv(n_vars); std::vector<Argument> spa(n_p);
        for (int i=0;i<n_vars;i++) { double v=py_variables[i].cast<double>(); sv[i]=(v==v)?v:0.0; }
        for (int k=0;k<n_p;k++) { sv[pidx[k]]=py_param_values[k].cast<double>(); spa[k]=sv[pidx[k]].markAsInput(); }
        idouble st(0.0); Argument sta = st.markAsInput();
        std::vector<idouble> sj(n); std::vector<Argument> sja(n);
        for (int i=0;i<n;i++) { sj[i]=0.0; sja[i]=sj[i].markAsInput(); }
        // Call compute_rates within step kernel recording
        py::list spkx(n),spkr(n),spkv(n_vars);
        for (int i=0;i<n;i++) spkx[i]=py::cast(sx[i]);
        for (int i=0;i<n;i++) spkr[i]=py::cast(idouble(0.0));
        for (int i=0;i<n_vars;i++) spkv[i]=py::cast(sv[i]);
        compute_rates_fn(py::cast(st),spkx,spkr,spkv);
        // Semi-implicit: x_new[i] = x[i] + dt*f[i] / (1 - dt*J[i])
        std::vector<Result> sxr(n);
        for (int i=0;i<n;i++) {
            auto o=spkr[i]; idouble fi=py::isinstance<idouble>(o)?o.cast<idouble>():idouble(o.cast<double>());
            idouble xnew = sx[i] + idouble(idt)*fi / (idouble(1.0) - idouble(idt)*sj[i]);
            sxr[i] = xnew.markAsOutput();
        }
        step_kernel->stopRecording();

        // ---- Record compute_variables kernel (for var observables) ----
        bool has_var_obs = false;
        std::vector<int> var_obs_indices; // which variable indices are needed
        for (auto item : py_obs_list) {
            auto t = item.cast<py::tuple>();
            if (t[0].cast<int>() == 1) { // kind=1 = var
                has_var_obs = true;
                var_obs_indices.push_back(t[2].cast<int>()); // var_raw_idx
            }
        }

        std::shared_ptr<Functions> cv_kernel;
        std::vector<Argument> cv_xa, cv_pa;
        std::vector<Result> cv_res;
        int n_cv_out = 0;

        std::shared_ptr<WorkSpace> cv_loop_ws;
        if (has_var_obs && !compute_variables_fn.is_none()) {
            // Deduplicate var indices
            std::sort(var_obs_indices.begin(), var_obs_indices.end());
            var_obs_indices.erase(std::unique(var_obs_indices.begin(), var_obs_indices.end()), var_obs_indices.end());
            n_cv_out = var_obs_indices.size();

            cv_kernel = std::make_shared<Functions>();
            cv_kernel->startRecording();
            // Same inputs as f kernel: x and p
            std::vector<idouble> cvx(n);
            cv_xa.resize(n);
            for (int i=0;i<n;i++) { cvx[i]=py_states[i].cast<double>(); cv_xa[i]=cvx[i].markAsInput(); }
            std::vector<idouble> cvv(n_vars);
            cv_pa.resize(n_p);
            for (int i=0;i<n_vars;i++) { double v=py_variables[i].cast<double>(); cvv[i]=(v==v)?v:0.0; }
            for (int k=0;k<n_p;k++) { cvv[pidx[k]]=py_param_values[k].cast<double>(); cv_pa[k]=cvv[pidx[k]].markAsInput(); }

            // Call compute_rates then compute_variables
            py::list pcvx(n), pcvr(n), pcvv(n_vars);
            for (int i=0;i<n;i++) pcvx[i]=py::cast(cvx[i]);
            for (int i=0;i<n;i++) pcvr[i]=py::cast(idouble(0.0));
            for (int i=0;i<n_vars;i++) pcvv[i]=py::cast(cvv[i]);
            compute_rates_fn(py::cast(idouble(0.0)), pcvx, pcvr, pcvv);
            compute_variables_fn(py::cast(idouble(0.0)), pcvx, pcvr, pcvv);

            // Mark needed var outputs
            cv_res.resize(n_cv_out);
            for (int i=0;i<n_cv_out;i++) {
                auto obj = pcvv[var_obs_indices[i]];
                idouble rv = py::isinstance<idouble>(obj) ? obj.cast<idouble>() : idouble(obj.cast<double>());
                cv_res[i] = rv.markAsOutput();
            }
            cv_kernel->stopRecording();
            cv_loop_ws = cv_kernel->createWorkSpace();
        }

        // ---- Record main tape ----
        auto mf = std::make_shared<Functions>();
        mf->startRecording();
        std::vector<idouble> x(n); std::vector<Argument> xa(n);
        for (int i=0;i<n;i++) { x[i]=py_states[i].cast<double>(); xa[i]=x[i].markAsInput(); }
        std::vector<idouble> pid2(n_p); std::vector<Argument> pa(n_p);
        for (int k=0;k<n_p;k++) { pid2[k]=py_param_values[k].cast<double>(); pa[k]=pid2[k].markAsInput(); }

        struct Obs{int kind,si,vri,op;double gt,sd,w,sc;};
        std::vector<Obs> obs;
        for (auto item:py_obs_list) { auto t=item.cast<py::tuple>();
            obs.push_back({t[0].cast<int>(),t[1].cast<int>(),t[2].cast<int>(),t[3].cast<int>(),
                           t[4].cast<double>(),t[5].cast<double>(),t[6].cast<double>(),t[7].cast<double>()}); }
        using Key=std::tuple<int,int,int>;
        struct Acc{idouble sum{0.0};int count=0;idouble mx,mn;bool init=false;};
        std::map<Key,Acc> accs;
        for (auto&o:obs) accs[{o.kind,o.si,o.op}]=Acc{};

        // Pre-allocate workspaces
        auto jac_ws = kernel->createWorkSpace();
        auto step_ws = step_kernel->createWorkSpace();
        auto loop_ws = kernel->createWorkSpace();  // for Newton f-kernel replays
        std::vector<double> dJ_val(n, 0.0);
        std::vector<std::vector<double>> cached_LU(n, std::vector<double>(n, 0.0));
        std::vector<int> cached_piv(n);
        for (int i=0;i<n;i++) { cached_piv[i]=i; cached_LU[i][i]=1.0; }

        int sc=0;
        int cp_interval = 100;
        for (int step=0;step<total_steps;step++) {
            for (int sub=0;sub<n_sub;sub++) {
                if (sc > 0 && sc % cp_interval == 0)
                    idouble::CheckPoint();

                double t_val = sc * idt;

                // Compute diagonal J off-tape via kernel reverse AD (every jac_lag steps)
                if (sc%jac_lag==0) {
                    for (int i=0;i<n;i++) jac_ws->setVal(kxa[i], x[i].val);
                    for (int k2=0;k2<n_p;k2++) jac_ws->setVal(kpa[k2], pid2[k2].val);
                    jac_ws->setVal(kta, t_val);
                    kernel->forward(*jac_ws);
                    for (int i=0;i<n;i++) {
                        jac_ws->resetDiff();
                        jac_ws->setDiff(krr[i], 1.0);
                        kernel->reverse(*jac_ws);
                        dJ_val[i] = aadc::toDblPtr(jac_ws->diff(kxa[i]))[0];
                    }
                }

                if (newton_iters == 0) {
                    // ---- SEMI-IMPLICIT: step kernel approach ----
                    // ONE reference node per step, no idouble arithmetic on main tape
                    std::vector<idouble> xnew(n);
                    std::vector<aadc::ExtVarIndex> txi(n), tpi(n_p), toi(n);
                    for (int i=0;i<n;i++) txi[i]=aadc::ExtVarIndex(x[i],true,true);
                    for (int k2=0;k2<n_p;k2++) tpi[k2]=aadc::ExtVarIndex(pid2[k2],true,true);
                    for (int i=0;i<n;i++){xnew[i]=idouble(0.0);toi[i]=aadc::ExtVarIndex(xnew[i],false,true);}

                    aadc::addConstStateExtFunction(std::make_shared<StepExtFunc>(
                        step_kernel, sxa, spa, sta, sja, sxr,
                        txi, tpi, toi, t_val, dJ_val, step_ws));

                    for (int i=0;i<n;i++) step_ws->setVal(sxa[i], x[i].val);
                    for (int k2=0;k2<n_p;k2++) step_ws->setVal(spa[k2], pid2[k2].val);
                    step_ws->setVal(sta, t_val);
                    for (int i=0;i<n;i++) step_ws->setVal(sja[i], dJ_val[i]);
                    step_kernel->forward(*step_ws);
                    for (int i=0;i<n;i++) xnew[i].val=aadc::toDblPtr(step_ws->val(sxr[i]))[0];
                    x = xnew;
                } else {
                    // ---- NEWTON: f-kernel + LUSolveExtFunc approach ----
                    // Semi-implicit predictor via step kernel
                    std::vector<idouble> y(n);
                    {
                        std::vector<aadc::ExtVarIndex> txi(n), tpi(n_p), toi(n);
                        for (int i=0;i<n;i++) txi[i]=aadc::ExtVarIndex(x[i],true,true);
                        for (int k2=0;k2<n_p;k2++) tpi[k2]=aadc::ExtVarIndex(pid2[k2],true,true);
                        for (int i=0;i<n;i++){y[i]=idouble(0.0);toi[i]=aadc::ExtVarIndex(y[i],false,true);}
                        aadc::addConstStateExtFunction(std::make_shared<StepExtFunc>(
                            step_kernel, sxa, spa, sta, sja, sxr,
                            txi, tpi, toi, t_val, dJ_val, step_ws));
                        for (int i=0;i<n;i++) step_ws->setVal(sxa[i], x[i].val);
                        for (int k2=0;k2<n_p;k2++) step_ws->setVal(spa[k2], pid2[k2].val);
                        step_ws->setVal(sta, t_val);
                        for (int i=0;i<n;i++) step_ws->setVal(sja[i], dJ_val[i]);
                        step_kernel->forward(*step_ws);
                        for (int i=0;i<n;i++) y[i].val=aadc::toDblPtr(step_ws->val(sxr[i]))[0];
                    }

                    // Full Jacobian + LU (off-tape, every jac_lag steps)
                    if (sc%jac_lag==0) {
                        std::vector<std::vector<double>> J(n, std::vector<double>(n, 0.0));
                        for (int i=0;i<n;i++) jac_ws->setVal(kxa[i], y[i].val);
                        for (int k2=0;k2<n_p;k2++) jac_ws->setVal(kpa[k2], pid2[k2].val);
                        jac_ws->setVal(kta, t_val);
                        kernel->forward(*jac_ws);
                        for (int i=0;i<n;i++) {
                            jac_ws->resetDiff();
                            jac_ws->setDiff(krr[i], 1.0);
                            kernel->reverse(*jac_ws);
                            for (int j=0;j<n;j++)
                                J[i][j] = aadc::toDblPtr(jac_ws->diff(kxa[j]))[0];
                        }
                        // LU factorize (I - dt*J)
                        cached_LU.assign(n, std::vector<double>(n));
                        cached_piv.resize(n);
                        for (int i=0;i<n;i++) { cached_piv[i]=i; for (int j=0;j<n;j++) cached_LU[i][j]=(i==j?1.0:0.0)-idt*J[i][j]; }
                        for (int col=0;col<n;col++) {
                            int mx=col; for (int r=col+1;r<n;r++) if (std::abs(cached_LU[r][col])>std::abs(cached_LU[mx][col])) mx=r;
                            if (mx!=col) { std::swap(cached_LU[col],cached_LU[mx]); std::swap(cached_piv[col],cached_piv[mx]); }
                            for (int r=col+1;r<n;r++) {
                                cached_LU[r][col]/=cached_LU[col][col];
                                for (int c=col+1;c<n;c++) cached_LU[r][c]-=cached_LU[r][col]*cached_LU[col][c];
                            }
                        }
                    }

                    // Newton iterations: f(y) via f-kernel + LUSolveExtFunc
                    for (int nit=0; nit<newton_iters; nit++) {
                        // f(y) on tape via KernelExtFunc
                        std::vector<idouble> fy(n);
                        {
                            std::vector<aadc::ExtVarIndex> txi(n),tpi(n_p),toi(n);
                            for (int i=0;i<n;i++) txi[i]=aadc::ExtVarIndex(y[i],true,true);
                            for (int k2=0;k2<n_p;k2++) tpi[k2]=aadc::ExtVarIndex(pid2[k2],true,true);
                            for (int i=0;i<n;i++){fy[i]=idouble(0.0);toi[i]=aadc::ExtVarIndex(fy[i],false,true);}
                            aadc::addConstStateExtFunction(std::make_shared<KernelExtFunc>(kernel,kxa,kpa,krr,txi,tpi,toi,loop_ws));
                            for (int i=0;i<n;i++) loop_ws->setVal(kxa[i],y[i].val);
                            for (int k2=0;k2<n_p;k2++) loop_ws->setVal(kpa[k2],pid2[k2].val);
                            loop_ws->setVal(kta, t_val);
                            kernel->forward(*loop_ws);
                            for (int i=0;i<n;i++) fy[i].val=aadc::toDblPtr(loop_ws->val(krr[i]))[0];
                        }
                        // F = y - x - dt*f(y)
                        std::vector<idouble> F(n);
                        for (int i=0;i<n;i++) F[i]=y[i]-x[i]-idouble(idt)*fy[i];
                        // LU solve via ExtFunc
                        std::vector<idouble> dy(n);
                        {
                            std::vector<aadc::ExtVarIndex> rhs_ei(n), out_ei(n);
                            for (int i=0;i<n;i++) rhs_ei[i]=aadc::ExtVarIndex(F[i],true,true);
                            for (int i=0;i<n;i++){dy[i]=idouble(0.0);out_ei[i]=aadc::ExtVarIndex(dy[i],false,true);}
                            aadc::addConstStateExtFunction(std::make_shared<LUSolveExtFunc>(n,cached_LU,cached_piv,rhs_ei,out_ei));
                            std::vector<double> bv(n);
                            for (int i=0;i<n;i++) bv[i]=F[cached_piv[i]].val;
                            for (int i=1;i<n;i++) for (int j=0;j<i;j++) bv[i]-=cached_LU[i][j]*bv[j];
                            for (int i=n-1;i>=0;i--) { for (int j=i+1;j<n;j++) bv[i]-=cached_LU[i][j]*bv[j]; bv[i]/=cached_LU[i][i]; }
                            for (int i=0;i<n;i++) dy[i].val=bv[i];
                        }
                        for (int i=0;i<n;i++) y[i]=y[i]-dy[i];
                    }
                    x = y;
                }
                sc++;
            }
            if (step>=pre_steps) {
                // Compute var observables via cv_kernel if needed
                std::vector<idouble> var_vals;
                if (has_var_obs && cv_kernel) {
                    var_vals.resize(n_cv_out);
                    std::vector<aadc::ExtVarIndex> cvtxi(n), cvtpi(n_p), cvtoi(n_cv_out);
                    for (int i=0;i<n;i++) cvtxi[i]=aadc::ExtVarIndex(x[i],true,true);
                    for (int k=0;k<n_p;k++) cvtpi[k]=aadc::ExtVarIndex(pid2[k],true,true);
                    for (int i=0;i<n_cv_out;i++){var_vals[i]=idouble(0.0);cvtoi[i]=aadc::ExtVarIndex(var_vals[i],false,true);}
                    aadc::addConstStateExtFunction(std::make_shared<KernelExtFunc>(cv_kernel,cv_xa,cv_pa,cv_res,cvtxi,cvtpi,cvtoi,cv_loop_ws));
                    for (int i=0;i<n;i++) cv_loop_ws->setVal(cv_xa[i],x[i].val);
                    for (int k=0;k<n_p;k++) cv_loop_ws->setVal(cv_pa[k],pid2[k].val);
                    cv_kernel->forward(*cv_loop_ws);
                    for (int i=0;i<n_cv_out;i++) var_vals[i].val=aadc::toDblPtr(cv_loop_ws->val(cv_res[i]))[0];
                }

                for (auto&o:obs) {
                    auto&a=accs[{o.kind,o.si,o.op}];
                    idouble xi;
                    if (o.kind==0) {
                        xi=x[o.si];
                    } else if (has_var_obs && cv_kernel) {
                        // Find var_obs_indices position for o.vri
                        auto it = std::find(var_obs_indices.begin(), var_obs_indices.end(), o.vri);
                        if (it != var_obs_indices.end()) xi = var_vals[it - var_obs_indices.begin()];
                        else continue;
                    } else continue;
                    switch(o.op){
                        case 0:a.sum=a.sum+xi;a.count++;break;
                        case 1:if(!a.init){a.mx=xi;a.init=true;}else a.mx=iIf(xi>a.mx,xi,a.mx);break;
                        case 2:if(!a.init){a.mn=xi;a.init=true;}else a.mn=iIf(xi<a.mn,xi,a.mn);break;
                        case 3:if(!a.init){a.mx=xi;a.mn=xi;a.init=true;}else{
                            a.mx=iIf(xi>a.mx,xi,a.mx);a.mn=iIf(xi<a.mn,xi,a.mn);}break;
                    }
                }
            }
        }
        idouble cost(0.0);
        for (auto&o:obs) {
            auto&a=accs[{o.kind,o.si,o.op}]; idouble ov;
            switch(o.op){case 0:ov=a.sum/idouble((double)a.count);break;case 1:ov=a.mx;break;
                case 2:ov=a.mn;break;case 3:ov=a.mx-a.mn;break;default:continue;}
            idouble res=(ov-idouble(o.gt))/idouble(o.sd);
            cost=cost+idouble(o.sc*o.w)*res*res;
        }
        Result cr=cost.markAsOutput();
        mf->stopRecording();

        cache.funcs=mf; cache.ws=mf->createWorkSpace();
        cache.x_args=xa; cache.p_args=pa; cache.cost_res=cr;
        cache.total_subs=total_subs;
        cache.newton_mode=newton_iters;
    }

    // ---- Evaluate ----
    for (int i=0;i<n;i++) cache.ws->setVal(cache.x_args[i],py_states[i].cast<double>());
    for (int k=0;k<n_p;k++) cache.ws->setVal(cache.p_args[k],py_param_values[k].cast<double>());
    cache.funcs->forward(*cache.ws);
    double cv=aadc::toDblPtr(cache.ws->val(cache.cost_res))[0];
    cache.ws->resetDiff();
    cache.ws->setDiff(cache.cost_res,1.0);
    cache.funcs->reverse(*cache.ws);
    py::list grad;
    for (int k=0;k<n_p;k++) grad.append(aadc::toDblPtr(cache.ws->diff(cache.p_args[k]))[0]);
    return py::make_tuple(cv,grad);
}

// Batched evaluation: 4 parameter sets in one AVX forward+reverse
py::tuple bdf_evaluate_batch(
    py::list py_states,
    py::list py_param_values_batch,  // list of 4 param lists
    int n_p
) {
    if (!cache.funcs) throw std::runtime_error("Call bdf_record_and_evaluate first to build tape");
    constexpr int NLANES = sizeof(mmType) / sizeof(double);
    int batch = py::len(py_param_values_batch);
    if (batch > NLANES) batch = NLANES;
    int n = py::len(py_states);

    // Set inputs: states same for all lanes, params different per lane
    for (int i = 0; i < n; i++) {
        mmType v;
        double sv = py_states[i].cast<double>();
        for (int lane = 0; lane < NLANES; lane++)
            aadc::toDblPtr(v)[lane] = sv;
        cache.ws->setVal(cache.x_args[i], v);
    }
    for (int k = 0; k < n_p; k++) {
        mmType v;
        for (int lane = 0; lane < NLANES; lane++) {
            if (lane < batch) {
                py::list pv = py_param_values_batch[lane].cast<py::list>();
                aadc::toDblPtr(v)[lane] = pv[k].cast<double>();
            } else {
                py::list pv = py_param_values_batch[0].cast<py::list>();
                aadc::toDblPtr(v)[lane] = pv[k].cast<double>();
            }
        }
        cache.ws->setVal(cache.p_args[k], v);
    }

    cache.funcs->forward(*cache.ws);

    py::list costs;
    for (int lane = 0; lane < batch; lane++)
        costs.append(aadc::toDblPtr(cache.ws->val(cache.cost_res))[lane]);

    // Reverse for each lane separately (adjoint is lane-specific)
    py::list all_grads;
    for (int lane = 0; lane < batch; lane++) {
        cache.ws->resetDiff();
        mmType d_cost;
        for (int l = 0; l < NLANES; l++) aadc::toDblPtr(d_cost)[l] = (l == lane) ? 1.0 : 0.0;
        cache.ws->setDiff(cache.cost_res, d_cost);
        cache.funcs->reverse(*cache.ws);
        py::list grad;
        for (int k = 0; k < n_p; k++)
            grad.append(aadc::toDblPtr(cache.ws->diff(cache.p_args[k]))[lane]);
        all_grads.append(grad);
    }
    return py::make_tuple(costs, all_grads);
}

void init_bdf_loop(py::module& m) {
    m.def("bdf_record_and_evaluate",&bdf_record_and_evaluate,
        py::arg("compute_rates_fn"),py::arg("states"),py::arg("variables"),
        py::arg("param_indices"),py::arg("param_values"),
        py::arg("total_steps"),py::arg("pre_steps"),py::arg("n_sub"),py::arg("idt"),
        py::arg("obs_list"),py::arg("compute_variables_fn")=py::none(),py::arg("jac_lag")=10,
        py::arg("newton_iters")=0, py::arg("jac_coloring")=py::list());
    m.def("bdf_evaluate_batch",&bdf_evaluate_batch,
        py::arg("states"),py::arg("param_values_batch"),py::arg("n_p"),
        "Evaluate 4 parameter sets simultaneously via AVX (requires prior bdf_record_and_evaluate call)");
    m.def("bdf_save_tape", &bdf_save_tape, py::arg("path"),
        "Save recorded tape to disk (boost serialization). Returns True on success.");
    m.def("bdf_load_tape", &bdf_load_tape, py::arg("path"),
        "Load tape from disk. Returns True on success. Subsequent bdf_record_and_evaluate uses cached tape.");
}
