#!/bin/bash
# Prints the running games, if any: Roblox (sober), any Steam game (its
# `reaper SteamLaunch` wrapper, or a binary under steamapps/common). The Steam
# client alone is not a game. Shells and greps whose own command line holds
# these words are not counted.
pgrep -fa 'sober|SteamLaunch|steamapps/common|Roblox' |
  grep -vE '^[0-9]+ (/usr/bin/|/bin/)?(zsh|bash|sh|grep|pgrep|flock|timeout) ' | tr '\n' ';'
