# Optimizing Pest Control Treatment to Minimize Harm to Predating Species

## Goal
This report applies optimal control to pesticide treatment of a predator/prey population where the prey is a pest. The goal is to minimize the pest population and potential harms to its predators.

## Methods
- _Mathematical_: I cast an optimal control problem by defining an ODE for pest and predator population growth, setting appropriate constraints on pesticide concentration via the control, assuming initial conditions and parameter constraints, and finally constructing the Hamiltonian. Using the Hamiltonian, I calculated the adjoint equation, transversality condition, and the optimality condition (necessary conditions). This system uses an isoperimetric constraint to bound the total amount of pestcide used, u(t). Solving requires introducing another state variable that replaces this constraint, which produces a state equation with two fixed endpoints.
- _Computational_: I used a more complex approach in the code, the secant method, since the RK4 forward-backward sweep cannot account for the dual-fixed endpoints on the state equation.

## Repo Guide
In this repo you'll find MATLAB code in the "MATLAB.m" file, while the .pdf file is the full typeset report. 
