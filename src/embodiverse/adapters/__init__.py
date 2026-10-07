"""Adapters: robots and simulators behind the robot port.

Each adapter implements `embodiverse.contracts.robot.Robot` for one robot
model or simulator, and converts between the vendor's or simulator's formats
and the conventions at its boundary. Adapters depend only on the contracts
and their own SDK or simulator (see the layers in docs/architecture.md), so
code above them runs unchanged on any of them.

`base` implements the port's semantics once: an adapter subclasses
`RobotBase`, or gives `TransportRobot` a `Transport` for its robot's wire,
and fills in only how state arrives and how commands are sent.
"""
