# Changelog

## [0.5.1](https://github.com/madkoding/nanobot/compare/v0.5.0...v0.5.1) (2026-08-15)


### Bug Fixes

* **telegram,channels:** reasoning delta guard, watchdog liveness on outbound, dead code cleanup ([#32](https://github.com/madkoding/nanobot/issues/32)) ([cb121eb](https://github.com/madkoding/nanobot/commit/cb121ebb73cf07962e052fa7405b6528cdce71b6))

## [0.5.0](https://github.com/madkoding/nanobot/compare/v0.4.0...v0.5.0) (2026-08-15)


### Features

* **telegram:** generative UI — rich messages, drafts nativos, reply keyboards, comandos dinámicos y ephemeral ([#22](https://github.com/madkoding/nanobot/issues/22)) ([51585b1](https://github.com/madkoding/nanobot/commit/51585b1e4005b71ccba95f20d577879d1f21229c))


### Bug Fixes

* **channels:** keep outbound dispatcher alive on stale durable messages ([#28](https://github.com/madkoding/nanobot/issues/28)) ([d8a06d8](https://github.com/madkoding/nanobot/commit/d8a06d8ee323c6bd434517af4abcd2d81c74061f))
* **owner:** full WhatsApp owner access + normalized sender match ([#30](https://github.com/madkoding/nanobot/issues/30)) ([5658f55](https://github.com/madkoding/nanobot/commit/5658f5543ba477f6f6c5c77c98386fb4a5d80072))
* **telegram:** cancel typing indicators in stop() via TypingIndicator.stop_all() ([#29](https://github.com/madkoding/nanobot/issues/29)) ([16476a6](https://github.com/madkoding/nanobot/commit/16476a6d7e2486bcd093b40b45b4c051d00fa10e))

## [0.4.0](https://github.com/madkoding/nanobot/compare/v0.3.12...v0.4.0) (2026-08-15)


### Features

* **agent:** repetition/loop detection ([#20](https://github.com/madkoding/nanobot/issues/20)) ([c99b6d6](https://github.com/madkoding/nanobot/commit/c99b6d6fb7b95e98adb201934db987ee5ed41eaf))
* **exec:** apply bwrap sandbox only on Linux restricted workspaces ([#24](https://github.com/madkoding/nanobot/issues/24)) ([5c313c4](https://github.com/madkoding/nanobot/commit/5c313c443f5cba78ea3023e7012d1933f67a2be6))
* **memory,tools:** owner-only Dream consolidation, workspace memory isolation, and owner-only sensitive tools ([#27](https://github.com/madkoding/nanobot/issues/27)) ([b5462f3](https://github.com/madkoding/nanobot/commit/b5462f3b0b0ff253d07c332b6b0aed860b4a78dd))
* **webui:** check updates from GitHub instead of PyPI ([#9](https://github.com/madkoding/nanobot/issues/9)) ([8c12630](https://github.com/madkoding/nanobot/commit/8c12630b824106a78d16b2469fa828f3845fb850))


### Bug Fixes

* **command:** restore /stop reply and gateway-restart notices ([2db36a4](https://github.com/madkoding/nanobot/commit/2db36a4a87a00efff7242d68e24398652f0e7430))
* **command:** restore /stop reply and gateway-restart notices ([709e20a](https://github.com/madkoding/nanobot/commit/709e20a7b6bdbf0f2536917ef4a4a5568eba0866))
* **loop:** ack inline-dispatched commands and runtime-control messages ([#15](https://github.com/madkoding/nanobot/issues/15)) ([338915b](https://github.com/madkoding/nanobot/commit/338915b89033ae84cc1558e4f3f8a99fa1297b9c))
* stop deleted/finished sessions resurrecting after gateway restart ([7a4cac6](https://github.com/madkoding/nanobot/commit/7a4cac680cb3afb1d2a66204b0b9f9ad6fbc22b5))
* stop deleted/finished sessions resurrecting after gateway restart ([27a7bbf](https://github.com/madkoding/nanobot/commit/27a7bbf03fa87b40862ae0f5a22ccbb2d993a0b7))
* **tests:** make onboard TTY fixture selectable and Windows-safe project context paths ([b195f58](https://github.com/madkoding/nanobot/commit/b195f587fe59c6d11f0d693dea0011b21cb1e7e4))
* **webui:** open projects list without leaking chat session key ([#10](https://github.com/madkoding/nanobot/issues/10)) ([2c758e7](https://github.com/madkoding/nanobot/commit/2c758e767ec12057d45885ac3853c4e6c43ce7d7))
* **webui:** stop infinite reload loop when opening a todo list ([3363e07](https://github.com/madkoding/nanobot/commit/3363e07f76e3b5b3f571ff83e6923e2d168bfabd))
* **webui:** stop infinite reload loop when opening a todo list ([328f417](https://github.com/madkoding/nanobot/commit/328f417772cb1e1cb6ad1ac3a8c97f7c23c2160b))
* **webui:** stop infinite reload loop when opening a todo list ([#12](https://github.com/madkoding/nanobot/issues/12)) ([3363e07](https://github.com/madkoding/nanobot/commit/3363e07f76e3b5b3f571ff83e6923e2d168bfabd))
* **whatsapp:** auto-reconnect when websocket dies silently ([#19](https://github.com/madkoding/nanobot/issues/19)) ([2ce738c](https://github.com/madkoding/nanobot/commit/2ce738c39af976ad532fbcadc3c13a24fc067f36))
