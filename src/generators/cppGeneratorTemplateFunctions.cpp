/* Functions for index-name mapping of states and variables */
std::string generateHeaderKey(const Model0d::VariableInfo& info, const std::vector<std::string>& modules)
{
    /*
    Replicates the logic in openOutputFiles to generate the exact string used in the output file headers.
    */
    std::string mod_suffix = "_module";
    std::string mod_name = std::string(info.component);
    std::string var_name = std::string(info.name);

    // Handle "parameters" component specifically (for EXTERNAL variable type only)
    if (mod_name == "parameters") {
        std::string detected_mod = "";
        for (const auto& m : modules) {
            if (var_name.find(m) != std::string::npos) {
                detected_mod = m;
                break;
            }
        }
        
        if (!detected_mod.empty()) {
            std::string var_suffix = "_" + detected_mod;
            size_t pos = var_name.rfind(var_suffix);
            if (pos != std::string::npos && pos == var_name.size() - var_suffix.size()) {
                var_name.erase(pos);
            }
            return detected_mod + "/" + var_name;
        }
        return "parameters/" + var_name; // Fallback
    }

    // Handle standard modules (strip _module)
    size_t pos = mod_name.rfind(mod_suffix);
    if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
        mod_name.erase(pos);
    }
    
    return mod_name + "/" + var_name;
}

std::string extractNameFromHeader(std::string item) 
{
    /*
    Extract module and variable name from the file header, e.g., "1: heart/q_ra[m3];" -> "heart/q_ra"
    */

    // Remove the index prefix "1: "
    size_t colonPos = item.find(':');
    if (colonPos != std::string::npos) {
        item = item.substr(colonPos + 1);
    }
    // Remove the units "[m3]"
    size_t bracketPos = item.find('[');
    if (bracketPos != std::string::npos) {
        item = item.substr(0, bracketPos);
    }
    // Trim whitespace
    item.erase(0, item.find_first_not_of(" \t\r\n"));
    item.erase(item.find_last_not_of(" \t\r\n") + 1);
    return item;
}
/* End functions */

/* Functions for import/export of model outputs */
void Model0d::openOutputFiles(std::string outDir)
{
    int status;
	const char * aux;
	aux = outDir.c_str();
	status = mkdir(aux, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (status != 0){
        std::cout << "0d solver :: openOutputFiles() could not create folder " << outDir.c_str() << " " << status << std::endl;
        perror( "Error opening file" );
        printf( "Error opening file: %s\n", strerror( errno ) );
    }

	std::string path = outDir+"sol0D_states.txt";
	outFileStates.open(path.c_str());
    path = outDir+"sol0D_variables.txt";
	outFileVars.open(path.c_str());

    int count = 0;
    std::vector<std::string> modules_list_tmp;
    std::string mod_suffix = "_module";
    outFileStates << "# 0: t[s]; ";
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        std::string mod_name;
        std::string state_name;
        std::string units;

        idx_states_to_output.push_back(i);
        
        mod_name = std::string(STATE_INFO[i].component);
        size_t pos = mod_name.rfind(mod_suffix);
        if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
            mod_name.erase(pos);
            if (std::find(modules_list_tmp.begin(), modules_list_tmp.end(), mod_name) == modules_list_tmp.end()) {
                modules_list_tmp.push_back(mod_name);
            }
        }
        state_name = std::string(STATE_INFO[i].name);
        units = std::string(STATE_INFO[i].units);
        std::string state_string = std::to_string(count+1) + ": " 
                                    + mod_name + "/" 
                                    + state_name + "[" 
                                    + units + "]; ";
        outFileStates << state_string;
        count++;
    }
    outFileStates << std::endl;
    outFileStates.flush();

    count = 0;
    outFileVars << "# 0: t[s]; ";
    for (size_t i = 0; i < VARIABLE_COUNT; ++i) {
    
        if (VARIABLE_INFO[i].type == EXTERNAL || VARIABLE_INFO[i].type == ALGEBRAIC){
        
            std::string mod_name;
            std::string var_name;
            std::string units;

            if (std::string(VARIABLE_INFO[i].component)=="parameters"){
                var_name = std::string(VARIABLE_INFO[i].name);
                for (size_t j = 0; j < modules_list_tmp.size(); ++j) {
                    if (var_name.find(modules_list_tmp[j]) != std::string::npos) {
                        mod_name = modules_list_tmp[j];
                        break;
                    }
                }
                std::string var_suffix = "_"+mod_name;
                size_t pos = var_name.rfind(var_suffix);
                if (pos != std::string::npos && pos == var_name.size() - var_suffix.size()) {
                    var_name.erase(pos);
                }
                units = std::string(VARIABLE_INFO[i].units);
                std::string var_string = std::to_string(count+1) + ": " 
                                        + mod_name + "/" 
                                        + var_name + "[" 
                                        + units + "]; ";
                outFileVars << var_string;
                idx_vars_to_output.push_back(i);
                count++;
            } else{
                mod_name = std::string(VARIABLE_INFO[i].component);
                size_t pos = mod_name.rfind(mod_suffix);
                if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
                    mod_name.erase(pos); 
                    var_name = std::string(VARIABLE_INFO[i].name);
                    units = std::string(VARIABLE_INFO[i].units);
                    std::string var_string = std::to_string(count+1) + ": " 
                                            + mod_name + "/" 
                                            + var_name + "[" 
                                            + units + "]; ";
                    outFileVars << var_string;
                    idx_vars_to_output.push_back(i);
                    count++;
                }
            }
        }
    }
    outFileVars << std::endl;
    outFileVars.flush();
}

void Model0d::loadFromFile(const std::string& filename, double* targetArray, 
                            const Model0d::VariableInfo infoArray[], size_t infoSize,
                            bool useFirstRow) 
{
    std::ifstream file(filename);
    if (!file.is_open()) return;

    std::string headerLine;
    if (!std::getline(file, headerLine)) return;

    // Build the list of all modules for parameter mapping
    std::vector<std::string> modules;
    std::string mod_suffix = "_module";
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        std::string m = std::string(STATE_INFO[i].component);
        size_t pos = m.rfind(mod_suffix);
        if (pos != std::string::npos) {
            m.erase(pos);
            if (std::find(modules.begin(), modules.end(), m) == modules.end()) modules.push_back(m);
        }
    }

    // Build Lookup Map
    std::map<std::string, int> lookup;
    for (size_t i = 0; i < infoSize; ++i) {
        lookup[generateHeaderKey(infoArray[i], modules)] = static_cast<int>(i);
    }

    // Parse Header Columns
    std::vector<int> colToArrayIndex;
    std::stringstream ss(headerLine);
    std::string item;
    int colCounter = 0;
    while (std::getline(ss, item, ';')) {
        std::string cleaned = extractNameFromHeader(item);
        if (cleaned.find("t[s]") != std::string::npos || cleaned == "t" || cleaned == "0") {
            colToArrayIndex.push_back(-1); // "time" is not stored inside the statesLoc or varLoc arrays
            // std::printf("Column %2d | %-30s -> [SKIP: Time/VOI]\n", colCounter, cleaned.c_str());
        } else if (lookup.count(cleaned)) {
            colToArrayIndex.push_back(lookup[cleaned]); // match
            // std::printf("Column %2d | %-30s -> Internal Index: %d\n", colCounter, cleaned.c_str(), lookup[cleaned]);
        } else {
            colToArrayIndex.push_back(-2); // unknown: file header contains a variable name that doesn't exist in STATE_INFO or VARIABLE_INFO
            // std::printf("Column %2d | %-30s -> [WARNING: No Match Found]\n", colCounter, cleaned.c_str());
        }
        colCounter++;
    }

    // Select the Data Row (only first or last row for now)
    std::string dataLine;
    if (useFirstRow) {
        while (std::getline(file, dataLine)) {
            if (!dataLine.empty()) break;
        }
    } else {
        std::string currentLine;
        while (std::getline(file, currentLine)) {
            if (!currentLine.empty()) dataLine = currentLine;
        }
    }

    if (!dataLine.empty()) {
        std::stringstream dataSS(dataLine);
        double colVal;
        int colIdx = 0;
        int loadedCount = 0;
        while (dataSS >> colVal && colIdx < colToArrayIndex.size()) {
            int targetIdx = colToArrayIndex[colIdx];
            if (targetIdx >= 0) {
                targetArray[targetIdx] = colVal;
                loadedCount++;
            }
            colIdx++;
        }
        std::cout << "Successfully loaded " << loadedCount << " values from " << filename << std::endl;
    }
}

void Model0d::writeOutput(double voiLoc)
{
    outFileStates << std::scientific << std::setprecision(18) << voiLoc << " ";
    for (size_t i = 0; i < idx_states_to_output.size(); ++i) {
		outFileStates << std::scientific << std::setprecision(18) << states[idx_states_to_output[i]] << " ";
    }
	outFileStates << std::endl;
    outFileStates.flush();

    outFileVars << std::scientific << std::setprecision(18) << voiLoc << " ";
    for (size_t i = 0; i < idx_vars_to_output.size(); ++i) {
		outFileVars << std::scientific << std::setprecision(18) << variables[idx_vars_to_output[i]] << " ";
    }
	outFileVars << std::endl;
    outFileVars.flush();
}

void Model0d::closeOutputFiles()
{
    outFileStates.close();
    outFileVars.close();
}
/* End functions */

/* Functions handling pipes */
int Model0d::openPipes(std::string pipePath)
{
    // Open write pipe for time step.
    std::string opipeName = pipePath+std::string("zero_to_parent_dt");
    write_pipe_dt.open(opipeName);
    if (!write_pipe_dt.is_open()) {
        std::cerr << "0d solver :: failed to open time step write pipe" << std::endl;
        return 1;
    }
    // Open write pipe for each 1d-0d connection.
    write_pipe.reserve(N1d0d);
    for (int i = 0; i < N1d0d; ++i) {
        std::string pipeID = std::to_string(i+1);
        std::ofstream ofs;
        std::string opipeName = pipePath+std::string("zero_to_parent_")+pipeID;
        // Open write pipe.
        ofs.open(opipeName);
        if (!ofs.is_open()) {
            std::cerr << "0d solver :: failed to open write pipe " << pipeID << std::endl;
            return 1;
        }  
        write_pipe.push_back(std::move(ofs));
    }
    // Open read pipe for time step.
    std::string ipipeName = pipePath+std::string("parent_to_zero_dt");
    read_pipe_dt.open(ipipeName);
    if (!read_pipe_dt.is_open()) {
        std::cerr << "0d solver :: failed to open time step read pipe" << std::endl;
        return 1;
    }
    // Open read pipe for each 1d-0d connection.
    read_pipe.reserve(N1d0d);
    for (int i = 0; i < N1d0d; ++i) {
        std::string pipeID = std::to_string(i+1);
        std::ifstream ifs;
        std::string ipipeName = pipePath+std::string("parent_to_zero_")+pipeID;
        // Open read pipe.
        ifs.open(ipipeName);
        if (!ifs.is_open()) {
            std::cerr << "0d solver :: failed to open read pipe " << pipeID << std::endl;
            return 1;
        }
        read_pipe.push_back(std::move(ifs));
    }

    DATA_LENGTH = 2;
    zero_data_dt = new double[DATA_LENGTH];
    parent_data_dt = new double[DATA_LENGTH];
    for (int j = 0; j < DATA_LENGTH; ++j){
        zero_data_dt[j] = 0.0;
        parent_data_dt[j] = 0.0;
    }
    zero_data = new double*[N1d0d];
    parent_data = new double*[N1d0d];
    for (int i = 0; i < N1d0d; ++i) {
        zero_data[i] = new double[DATA_LENGTH];
        parent_data[i] = new double[DATA_LENGTH];
    }
    for (int i = 0; i < N1d0d; ++i){
        for (int j = 0; j < DATA_LENGTH; ++j){
            zero_data[i][j] = 0.0;
            parent_data[i][j] = 0.0;
        }
    }

    // Open read pipe for volume sum.
    std::string ipipeName = pipePath+std::string("parent_to_zero_vol");
    read_pipe_vol.open(ipipeName);
    if (!read_pipe_vol.is_open()) {
        std::cerr << "0d solver :: failed to open volume sum read pipe" << std::endl;
        return 1;
    }
   
    return 0;
}

void Model0d::closePipes()
{
    // Close time step pipes
    write_pipe_dt.close();
    read_pipe_dt.close();
    // Close 1d-0d connection pipes
    for (int i = 0; i < N1d0d; ++i) {
        write_pipe[i].close();
        read_pipe[i].close();
    }
    // Close volume sum pipe
    read_pipe_vol.close();
 
    std::cout << "### 0d solver :: All pipes closed. ###" << std::endl;
}
/* End functions */