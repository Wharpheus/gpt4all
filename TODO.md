# TODO: Optimize Build Pipeline, Harden Security, Customize for Solidity

## Optimize Pipeline
- [x] Increase job parallelism in CircleCI config (added new workflow)
- [x] Upgrade resource classes to larger sizes where possible
- [ ] Add more granular caching strategies

## Harden Security
- [x] Add CodeQL job for static application security testing (SAST)
- [x] Add Snyk job for dependency vulnerability scanning
- [x] Add security hardening compiler flags
- [ ] Add security tests (e.g., fuzzing, penetration testing)

## Customize for Solidity
- [ ] Add Solidity compilation job using solc
- [ ] Add Slither job for Solidity static analysis
- [ ] Add Mythril job for Solidity security analysis
- [ ] Integrate GPT4All for AI-assisted Solidity contract fixing suggestions

## Implementation Steps
- [x] Edit .circleci/continue_config.yml to add new jobs and workflows
- [ ] Test pipeline changes
- [ ] Update documentation if needed
