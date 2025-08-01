# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added


### Changed


### Fixed


### Removed


## [v0.19.0] - 2025-07-16

### Added

- Added simplified conditional documentation using the `#[enable_doc_switch]` macro (#3630)
- `error!` and `warning!` (moved from `esp-build`) (#3645)
- Added `doc_replace` macro (#3744)

### Changed

- MSRV is now 1.88.0 (#3742)
- The `handler` macro no longer accepts priority as a string (#3643)

## [v0.18.0] - 2025-06-03

### Changed

- Using the `#handler` macro with a priority of `None` will fail at compile time (#3304)
- Bump Rust edition to 2024, bump MSRV to 1.86. (#3391, #3560)

## [0.17.0] - 2025-02-24

### Added

- Added `#[builder_lite(into)]` attribute that generates the setter with `impl Into<T>` parameter (#3011)
- Added `#[builder_lite(skip_setter)]` and `#[builder_lite(skip_getter)]` attribute to skip generating setters or getters (#3011)
- Added `#[builder_lite(skip)]` to ignore a field completely (#3011)

## 0.16.0 - 2025-01-15

### Added

- Added the `BuilderLite` derive macro which implements the Builder Lite pattern for a struct (#2614)

### Changed

- Functions marked with `#[handler]` can now be referenced in `const` context. (#2559)
- Bump MSRV to 1.84 (#2951)

### Removed

- Removed the `enum-dispatch`, `interrupt`, and `ram` features (#2594)

## 0.15.0 - 2024-11-20

### Changed

- Remove `get_` prefix from functions (#2528)

## 0.14.0 - 2024-10-10

## 0.13.0 - 2024-08-29

## 0.12.0 - 2024-07-15

## 0.11.0 - 2024-06-04

## 0.10.0 - 2024-04-18

## 0.9.0 - 2024-03-18

## 0.8.0 - 2023-12-12

## 0.7.0 - 2023-10-31

## 0.6.1 - 2023-09-05

## 0.6.0 - 2023-07-04

## 0.5.0 - 2023-03-27

## 0.4.0 - 2023-02-21

## 0.2.0 - 2023-01-26

## 0.1.0 - 2022-08-25

### Added

- Initial release (#2518)

[0.17.0]: https://github.com/esp-rs/esp-hal/releases/tag/esp-hal-procmacros-v0.17.0
[v0.18.0]: https://github.com/esp-rs/esp-hal/compare/esp-hal-procmacros-v0.17.0...esp-hal-procmacros-v0.18.0
[v0.19.0]: https://github.com/esp-rs/esp-hal/compare/esp-hal-procmacros-v0.18.0...esp-hal-procmacros-v0.19.0
[Unreleased]: https://github.com/esp-rs/esp-hal/compare/esp-hal-procmacros-v0.19.0...HEAD
