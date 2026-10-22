# Remove preprocessor line markers from a flattened shader. They name host
# paths the Metal front end cannot open, and it reports diagnostics against
# them.
file(READ "${IN}" _text)
string(REGEX REPLACE "(^|\n)#[^\n]*" "\\1" _text "${_text}")
file(WRITE "${OUT}" "${_text}")
