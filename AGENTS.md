# AGENTS.md

## Repository expectations
* Keep an eye toward simple implementations with minimal code surface area
* Do not add excessive runtime checks that could instead be done with static typechecking (i.e., pyright)
* Avoid typing hacks like `cast()`
* Do not add imports to __init__. I will handle this myself
