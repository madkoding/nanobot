# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Input validation module (`nanobot.utils.validation`) with comprehensive validation functions:
  - `validate_session_key()` - Validates session identifiers (max 64 chars, alphanumeric + `_:-`)
  - `validate_chat_id()` - Validates chat IDs (max 128 chars)
  - `validate_sender_id()` - Validates sender IDs (max 128 chars)
  - `validate_channel()` - Validates channel names (max 64 chars)
  - `sanitize_input()` - Sanitizes user messages (max 50k chars)
  - `validate_media_paths()` - Prevents path traversal attacks in media files
  - `ValidationError` exception class for validation failures
- Input validation integrated into all public SDK methods:
  - `Nanobot.run()` - validates all input parameters
  - `Nanobot.run_streamed()` - validates all input parameters
  - `Nanobot.stream()` - validates all input parameters
- Enhanced docstrings for all public methods with complete Args and Raises sections
- `ValidationError` exported from main `nanobot` module for easy access

### Changed
- Code formatting standardized across entire codebase using `ruff format` (370 files reformatted)
- All linting issues automatically fixed with `ruff check --fix`
- Improved documentation for context manager methods (`__aenter__`, `__aexit__`)

### Security
- **Breaking**: Session keys with invalid characters now raise `ValidationError` instead of potentially causing undefined behavior
- **Breaking**: Messages exceeding 50,000 characters now raise `ValidationError`
- **Breaking**: Media paths containing `..` components are rejected to prevent path traversal
- All user inputs to SDK methods are now validated before processing

### Fixed
- Code formatting inconsistencies across 190+ files
- Missing docstrings in public API methods

## [0.3.12] - Previous Version

[Unreleased]: https://github.com/nanobot-org/nanobot/compare/v0.3.12...HEAD
