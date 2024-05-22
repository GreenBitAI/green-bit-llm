# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.1.0] - 2024/01/05

### Added

- Integration with Bitorch Engine
- Full-parameter fine-tuning and PEFT support 
- Fast inference capabilities
- Comprehensive evaluation tools and detailed model evaluation results

## [0.2.0] - 2024/22/05

### Added

- Initial support for a classical GPTQ model using the MPQLinear layer
- AutoGPTQ information and commands in the repository
- Support for LoRA and GPTQ evaluation
- SFT comparison updates
- Missing comment to the customized trainer class

### Fixed

- Issue in GbaSFTTrainer for saving non-GBA models
- Mismatch issue between GPTQ and LoRA
- Bug preventing quant_strategy.json from being saved during SFT

### Updated

- README with new AutoGPTQ and GPTQ support information