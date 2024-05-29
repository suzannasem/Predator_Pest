# Predator Pest Optimal Control Project

This report applies optimal control to pesticide treatment of a predator/prey population where the prey is a pest. In this repo you'll find MATLAB code in the "MATLAB.m" file and the full typeset report. Note that the MATLAB code will not run unless you create a function file with the same name. 

This system uses an isoperimetric constraint to bound the total amount of pestciide used, u(t). Solving requires introducing another state variable that replaces this constraint, but this produces a state equation with two fixed endpoints. We used a more complex approach in our code, the secant method, since the RK4 forward-backward sweep cannot account for this.
