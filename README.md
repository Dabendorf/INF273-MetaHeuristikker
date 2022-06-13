# INF273 Meta heuristics
This project includes my solution for the UiB course [INF273 Meta heuristics]([https://uib.kattis.com/courses/INF237/spring22](https://www.uib.no/emne/INF273)), in which we learned to build different meta heuristics to solve various algorithmic problems.


# Aim
The aim of this project was to build a heuristic to solve a logistics problem. The input file consists of a set of cargos which needs to be transported from A to B. We are given a set of trucks having a capacity to transport these cargos.

The cargo information includes
* the capacity of the cargo
* the time window to pickup the cargo
* the time window to deliver the cargo
* the node numbers where to pickup and deliver

The traffic information includes
* the distance of time between two places for each vehicle
* the cost to drive from A to be for each vehicle

The truck information includes
* a list which orders can be transported by it
* the maximum capacity of the truck
* the home node of the truck
* the starting time of it
