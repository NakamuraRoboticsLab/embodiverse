"""The runtime: what runs a policy on a robot, tick by tick.

The runtime depends only on the contracts (docs/architecture.md). It finds
adapters, observation terms and components through the registry, by name,
and never imports them. `embodiverse.runtime.observations` builds a
policy's input from the robot's state, as the policy's descriptor says.
`embodiverse.runtime.policy` turns each observation into an action, and
`embodiverse.runtime.safety` checks every tick's inputs and filters every
command before it reaches the robot. `embodiverse.runtime.runner` runs the
loop that joins them, at the policy's rate, on a robot it is given, and
`embodiverse.runtime.recording` records runs, with a manifest of what
produced them.
"""
