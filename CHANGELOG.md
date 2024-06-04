# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.2.4] - 2024/06/04

### Fixed

- Source distribution (was missing `requirements.txt`)

## [0.2.3] - 2024/05/26

### Added

- Evaluation results

### Fixed

- Changelog order and date format
- URL in README for PyPI

## [0.2.2] - 2024/05/24

### Added

- Evaluation results

### Fixed

- Version numbering

## [0.2.1] - 2024/05/22

### Added

- Missing changelog entries

### Fixed

- Version numbering

## [0.2.0] - 2024/05/20

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

## [0.1.0] - 2024/01/05

### Added

- Integration with Bitorch Engine
- Full-parameter fine-tuning and PEFT support 
- Fast inference capabilities
- Comprehensive evaluation tools and detailed model evaluation results
