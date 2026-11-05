"""Fail a CI job whose tests were collected but skipped.

Every UQ test is guarded by a ``skipif`` on an optional extra -- ``[uq]`` for pymc, ``[emulation]``
for autoemulate. So if one of those fails to install, the job that exists to run them skips every
one of them and reports success: a green badge over a job that ran nothing. That is the likely
failure mode here, not a red X.

Usage::

    python .github/scripts/fail_if_tests_skipped.py junit-uq*.xml

Exits non-zero if any named file collected nothing, ran nothing, or skipped any test. Globs are
expanded here rather than by the shell, so a pattern that matches no file is reported as "no
results were produced" instead of being passed through literally and read as a missing file.
"""
import glob
import sys
import xml.etree.ElementTree as ET


def main(patterns):
    problems = []
    seen_any = False

    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            seen_any = True
            suites = ET.parse(path).getroot()
            suites = [suites] if suites.tag == 'testsuite' else suites.findall('testsuite')
            for suite in suites:
                total = int(suite.get('tests', 0))
                skipped = int(suite.get('skipped', 0))
                ran = total - skipped
                print(f'{path}: {total} collected, {skipped} skipped, {ran} ran')
                if total == 0:
                    problems.append(f'{path} collected no tests')
                elif ran == 0:
                    problems.append(f'{path} ran nothing -- all {total} tests skipped')
                elif skipped:
                    problems.append(f'{path} skipped {skipped} of {total} tests')

    if not seen_any:
        problems.append(f'no results were produced for {" ".join(patterns)}, '
                        'so no tests ran at all')

    if problems:
        print()
        print('This job is meant to be the one place these run. Every test here is behind a '
              'skipif on an optional extra, so a skip almost always means [uq] or [emulation] '
              'did not install rather than that the test was inapplicable.')
        for problem in problems:
            print(f'  - {problem}')
        return 1

    print('every test ran')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['junit-*.xml']))
