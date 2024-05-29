function predprey

% code for pred-prey lab in the text, with fixed endpoints for one state
% isoperimetric constraint becomes x3(t)

N = 1000;

% arbitrary initial guesses for theta->lambda1(T)
theta_start = -0.52;   % a
theta_stop = -0.5;   % b

%%%%%% INITIAL PARAMETERS %%%%%%%%%%%%%
d1 = 0.1; d2 = 0.1; M = 1; A = 1; T = 10;
s = 5; % target final state -- state(T) = s

a = theta_start;
b = theta_stop;

% ODE parameters (initial conditions for states)
x10 = 10;
x20 = 1;
x30 = 0;

y = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t = y(:,1);
X1 = y(:,2);
X2 = y(:,3);
X3 = y(:,4);
U = y(:,5);

figure(1)
subplot(2,1,1)
plot(t,X1,'LineWidth',2,'-',t,X2,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with Initial Parameters')
subplot(2,1,2)
plot(t,U,'LineWidth',2)
xlabel('Time')
ylabel('Application Rate')
title('Optimal Rate of Pesticide Application with Initial Parameters')

disp(['Secant Method for Figure 1 Completed.'])

%%%%%% Changing d2,n10,n20,B,T a,b %%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1; A = 1;

s = 1;
T = 5;

y1 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t1 = y1(:,1);
X11 = y1(:,2);
X21 = y1(:,3);
X31 = y1(:,4);
U1 = y1(:,5);

figure(2)
subplot(2,1,1)
plot(t1,X11,'-','LineWidth',2,t1,X21,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with Adjusted Parameters')
subplot(2,1,2)
plot(t1,U1,'LineWidth',2)
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with Adjusted Parameters')

disp(['Secant Method for Figure 2 Completed'])

%%%%%%% Changing A %%%%%%%%%%%
a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1; A = 10;

s = 1;
T = 5;

y2 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t2 = y2(:,1);
X12 = y2(:,2);
X22 = y2(:,3);
X32 = y2(:,4);
U2 = y2(:,5);

figure(3)
subplot(2,1,1)
plot(t2,X12,'LineWidth',2,'-',t2,X22,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted A')
subplot(2,1,2)
plot(t2,U2,'LineWidth',2, '-.','DisplayName', 'A = 10')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'A = 1')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted A')

disp(['Secant Method for Figure 3 Completed'])

%%%%%%%%%% Changing d1 %%%%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.3; d2 = 0.01; M = 1; A = 1;

s = 1;
T = 5;

y3 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t3 = y3(:,1);
X13 = y3(:,2);
X23 = y3(:,3);
X33 = y3(:,4);
U3 = y3(:,5);

figure(4)
subplot(2,1,1)
plot(t3,X13,'-','LineWidth',2,t3,X23,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted d_1')
subplot(2,1,2)
plot(t3,U3,'LineWidth',2, '-.','DisplayName', 'd_1 = 0.3')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'd_1 = 0.1')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted d_1')

disp(['Secant Method for Figure 4 Completed'])


%%%% Varying d2 %%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.03; M = 1; A = 1;

s = 1;
T = 5;

y4 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t4 = y4(:,1);
X14 = y4(:,2);
X24 = y4(:,3);
X34 = y4(:,4);
U4 = y4(:,5);

figure(5)
subplot(2,1,1)
plot(t4,X14,'-','LineWidth',2,t4,X24,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted d_2')
subplot(2,1,2)
plot(t4,U4,'LineWidth',2, '-.','DisplayName', 'd_2 = 0.03')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'd_2 = 0.01')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted d_2')

disp(['Secant Method for Figure 5 Completed'])

%%%%%%%%% Varying M %%%%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1.5; A = 1;

s = 1;
T = 5;

y5 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t5 = y5(:,1);
X15 = y5(:,2);
X25 = y5(:,3);
X35 = y5(:,4);
U5 = y5(:,5);

figure(6)
subplot(2,1,1)
plot(t5,X15,'-','LineWidth',2,t5,X25,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted M')
subplot(2,1,2)
plot(t5,U5,'LineWidth',2, '-.','DisplayName', 'M = 1.5')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'M = 1')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted M')

disp(['Secant Method for Figure 6 Completed'])

%%%%%%%%%%% Varying B %%%%%%%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1; A = 1;

s = 0.5;
T = 5;

y6 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t6 = y6(:,1);
X16 = y6(:,2);
X26 = y6(:,3);
X36 = y6(:,4);
U6 = y6(:,5);

figure(7)
subplot(2,1,1)
plot(t6,X16,'-','LineWidth',2,t6,X26,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted B')
subplot(2,1,2)
plot(t6,U6,'LineWidth',2,'-.','DisplayName','B = 0.5')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'B = 1')
hold off
lgd = legend('Location','southeast')
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted B')

disp(['Secant Method for Figure 7 Completed'])

%%%%%%%%% varying N10 %%%%%%%%%%%%%5

a = -0.2;
b = -0.18;

x10 = 6;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1; A = 1;

s = 1;
T = 5;

y7 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t7 = y7(:,1);
X17 = y7(:,2);
X27 = y7(:,3);
X37 = y7(:,4);
U7 = y7(:,5);

figure(8)
subplot(2,1,1)
plot(t7,X17,'-','LineWidth',2,t7,X27,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted N10')
subplot(2,1,2)
plot(t7,U7,'LineWidth',2,'-.','DisplayName', 'N10 = 6')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'N10 = 5')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted N10')

disp(['Secant Method for Figure 8 Completed'])

%%%%%%%%% Vary N20 %%%%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 1;

d1 = 0.1; d2 = 0.01; M = 1; A = 1;

s = 1;
T = 5;

y8 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t8 = y8(:,1);
X18 = y8(:,2);
X28 = y8(:,3);
X38 = y8(:,4);
U8 = y8(:,5);

figure(9)
subplot(2,1,1)
plot(t8,X18,'-','LineWidth',2,t8,X28,'LineWidth',2,'-.')
legend('Prey', 'Predator')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with adjusted N20')
subplot(2,1,2)
plot(t8,U8,'LineWidth',2,'-.','DisplayName','N20 = 1')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'N20 = 2')
hold off
lgd = legend('Location','southeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with adjusted N20')

disp(['Secant Method for Figure 9 Completed'])

%%%%%%%%%%%% Vary T %%%%%%%%%%%%%%%

a = -0.2;
b = -0.18;

x10 = 5;
x20 = 2;

d1 = 0.1; d2 = 0.01; M = 1; A = 1;

s = 1;
T = 10;

y9 = secant1(a,b,N,s,d1,d2,M,A,x10,x20,x30,T);

t9 = y9(:,1);
X19 = y9(:,2);
X29 = y9(:,3);
X39 = y9(:,4);
U9 = y9(:,5);

figure(10)
subplot(2,1,1)
plot(t9,X19,'-','LineWidth',2,t9,X29,'LineWidth',2,'-.')
legend('Prey', 'Predator', 'Location', 'southeast')
xlabel('Time')
ylabel('Population Size')
title('Predator-Prey Populations with T = 10')
subplot(2,1,2)
plot(t9,U9,'LineWidth',2,'-.','DisplayName', 'T = 10')
hold on
plot(t1,U1,'LineWidth',2,'DisplayName', 'T = 5')
hold off
lgd = legend('Location','northeast');
xlabel('Time')
ylabel('Application Rate')
title('Optimal Pesticide Application with T = 10')

disp(['Secant Method for Figure 10 Completed'])


%--------------------------------------------------------------------------

% secant method function
function y = secant1(theta,a,b,s,d1,d2,M,A,x10,x20,x30,T)

% time interval
t0 = 0;
N = 1000;

flag = -1; % on/off switch

% first forward-backward sweep
z = sweep(a,x10,x20,x30,t0,T,N,d1,d2,M,A);
Va = z(N+1,4) - s; % target value for x3(1) = s (third column of output matrix)
% second forward-backward sweep
z = sweep(b,x10,x20,x30,t0,T,N,d1,d2,M,A);
Vb = z(N+1,4) - s;

n = 2; % number of iterations for convergence (two calls to sweep(...) above)

while(flag < 0)
    if(abs(Va) > abs(Vb))
        % update the interval bounds for secant method
        k = a;
        a = b;
        b = k;
        k = Va;
        Va = Vb;
        Vb = k;
    end

    d = Va*(b - a)/(Vb - Va); % difference quotient*function_value
    b = a;
    Vb = Va;
    a = a - d;
    z = sweep(a,x10,x20,x30,t0,T,N,d1,d2,M,A);
    Va = z(N+1,4) - s;

    if(abs(Va) < 1E-10)
        flag = 1;
    else
        n = n+1;
        disp([num2str(n) ' iterations completed.'])
    end
end

y = z;



%--------------------------------------------------------------------------
% forward-backward sweep function
function y = sweep(theta,x10,x20,x30,t0,T,N,d1,d2,M,A)

test = -1;

% convergence parameters
tmin = t0; tmax = T;
delta = 0.001;
t = linspace(tmin,tmax,N+1);
time = t';

h = T/N;
h2 = h/2;

x1 = zeros(N+1,1);
x2 = zeros(N+1,1);
x3 = zeros(N+1,1);
z = zeros(N+1,1);
x1(1) = x10;
x2(1) = x20;
x3(1) = x30;

% transversality conditions with theta guess for fixed endpoint
lambda1 = zeros(N+1,1);
lambda1(N+1) = 1;
lambda2 = zeros(N+1,1);
lambda2(N+1) = 0;
lambda3 = zeros(N+1,1);
lambda3(N+1) = theta;

u = zeros(N+1,1);

% define ODEs
fx1 = @(t,x1,x2,x3,u) (1-x2).*x1 - d1.*x1.*u;
fx2 = @(t,x1,x2,x3,u) (x1-1).*x2 - d2.*x2.*u;
fx3 = @(t,x1,x2,x3,u) u;

fl1 = @(t,x1,x2,x3,u,l1,l2,l3) l1.*(x2-1) + l1.*d1.*u - l2.*x2;
fl2 = @(t,x1,x2,x3,u,l1,l2,l3) l1.*x1 + l2.*(1-x1) + l2.*d2.*u;
fl3 = @(t,x1,x2,x3,u,l1,l2,l3) 0;

while(test < 0)

    oldu = u;
    oldx1 = x1;
    oldx2 = x2;
    oldx3 = x3;
    oldlambda1 = lambda1;
    oldlambda2 = lambda2;
    oldlambda3 = lambda3;

    for i = 1:N
        u2 = 0.5*(u(i)+u(i+1));

        k11 = fx1(t(i),x1(i),x2(i),x3(i),u(i));
        k12 = fx2(t(i),x1(i),x2(i),x3(i),u(i));
        k13 = fx3(t(i),x1(i),x2(i),x3(i),u(i));

        k21 = fx1(t(i)+h2,x1(i)+h2*k11,x2(i)+h2*k12,x3(i)+h2*k13,u2);
        k22 = fx2(t(i)+h2,x1(i)+h2*k11,x2(i)+h2*k12,x3(i)+h2*k13,u2);
        k23 = fx3(t(i)+h2,x1(i)+h2*k11,x2(i)+h2*k12,x3(i)+h2*k13,u2);

        k31 = fx1(t(i)+h2,x1(i)+h2*k21,x2(i)+h2*k22,x3(i)+h2*k23,u2);
        k32 = fx2(t(i)+h2,x1(i)+h2*k21,x2(i)+h2*k22,x3(i)+h2*k23,u2);
        k33 = fx3(t(i)+h2,x1(i)+h2*k21,x2(i)+h2*k22,x3(i)+h2*k23,u2);

        k41 = fx1(t(i)+h,x1(i)+h*k31,x2(i)+h*k32,x3(i)+h*k33,u(i+1));
        k42 = fx2(t(i)+h,x1(i)+h*k31,x2(i)+h*k32,x3(i)+h*k33,u(i+1));
        k43 = fx3(t(i)+h,x1(i)+h*k31,x2(i)+h*k32,x3(i)+h*k33,u(i+1));

        x1(i+1) = x1(i) + h/6*(k11 + 2*k21 + 2*k31 + k41);
        x2(i+1) = x2(i) + h/6*(k12 + 2*k22 + 2*k32 + k42);
        x3(i+1) = x3(i) + h/6*(k13 + 2*k23 + 2*k33 + k43);
    end

    for i = 1:N
        j = N + 2 - i;
        u2 = 0.5*(u(j)+u(j-1));
        x12 = 0.5*(x1(j)+x1(j-1));
        x22 = 0.5*(x2(j)+x2(j-1));
        x32 = 0.5*(x3(j)+x3(j-1));

        k11 = fl1(t(j),x1(j),x2(j),x3(j),u(j),lambda1(j),lambda2(j),lambda3(j));
        k12 = fl2(t(j),x1(j),x2(j),x3(j),u(j),lambda1(j),lambda2(j),lambda3(j));
        k13 = fl3(t(j),x1(j),x2(j),x3(j),u(j),lambda1(j),lambda2(j),lambda3(j));

        k21 = fl1(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k11,lambda2(j)-h2*k12,lambda3(j)-h2*k13);
        k22 = fl2(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k11,lambda2(j)-h2*k12,lambda3(j)-h2*k13);
        k23 = fl3(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k11,lambda2(j)-h2*k12,lambda3(j)-h2*k13);

        k31 = fl1(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k21,lambda2(j)-h2*k22,lambda3(j)-h2*k23);
        k32 = fl2(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k21,lambda2(j)-h2*k22,lambda3(j)-h2*k23);
        k33 = fl3(t(j)-h2,x12,x22,x32,u2,lambda1(j)-h2*k21,lambda2(j)-h2*k22,lambda3(j)-h2*k23);

        k41 = fl1(t(j)-h,x1(j-1),x2(j-1),x3(j-1),u(j-1),lambda1(j)-h*k31,lambda2(j)-h*k32,lambda3(j)-h*k33);
        k42 = fl2(t(j)-h,x1(j-1),x2(j-1),x3(j-1),u(j-1),lambda1(j)-h*k31,lambda2(j)-h*k32,lambda3(j)-h*k33);
        k43 = fl3(t(j)-h,x1(j-1),x2(j-1),x3(j-1),u(j-1),lambda1(j)-h*k31,lambda2(j)-h*k32,lambda3(j)-h*k33);

        lambda1(j-1) = lambda1(j) - (h/6)*(k11 + 2*k21 + 2*k31 + k41);
        lambda2(j-1) = lambda2(j) - (h/6)*(k12 + 2*k22 + 2*k32 + k42);
        lambda3(j-1) = lambda3(j) - (h/6)*(k13 + 2*k23 + 2*k33 + k43);
    end

    u1 = min(M,max((1/A).*(lambda1.*d1.*x1 + lambda2.*d2.*x2 - lambda3),0));
    u = 0.5*(u1 + oldu);

    temp1 = delta*sum(abs(u)) - sum(abs(oldu - u));
    temp2 = delta*sum(abs(x1)) - sum(abs(oldx1 - x1));
    temp3 = delta*sum(abs(x2)) - sum(abs(oldx2 - x2));
    temp4 = delta*sum(abs(x3)) - sum(abs(oldx3 - x3));
    temp5 = delta*sum(abs(lambda1)) - sum(abs(oldlambda1 - lambda1));
    temp6 = delta*sum(abs(lambda2)) - sum(abs(oldlambda2 - lambda2));
    temp7 = delta*sum(abs(lambda3)) - sum(abs(oldlambda3 - lambda3));
    test = min([temp1, temp2,temp3,temp4,temp5,temp6,temp7]);
end

y(:,1) = time;
y(:,2) = x1;
y(:,3) = x2;
y(:,4) = x3; % <- OUTPUT NEEDED FOR SECANT METHOD
y(:,5) = u;
