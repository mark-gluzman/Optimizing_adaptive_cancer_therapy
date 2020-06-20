
%% The code partially reproduces Figure 3 from paper
% Optimizing adaptive cancer therapy: dynamic programming and evolutionary game theory,
% Proceedings of the Royal Society B: Biological Sciences 287: 20192454 (2020)
% https://doi.org/10.1098/rspb.2019.2454

% Mark Gluzman, Jacob G. Scott, and Alexander Vladimirsky


% Produce optimal control in feedback form and the value function
% for a model of lung cancer proposed by Kaznatcheev et al. [1]
function   Figure3()
    
    %% Model parameters
    d_max = 3; % MTD
    sigma = 0.01; % time penalty
    ba = 2.5; % the benefit per unit of acidification
    bv = 2; % the benefit from the oxygen per unit of vascularization
    c = 1; % the cost of production VEGF
    n_neigh = 4; %the number of cells in the interaction group.
    fb = 10^(-1.5); % failure barrier, recovery barrier
    
    %% Discretization parameters
    n = 9000; % number of meshpoints along one side
    h = 1 ./ n;
    % the algorithm terminates when the difference between value functions
    % in sequential iterations falls below 'iter_tol'
    iter_tol = 10^(-4);
    
    hugeVal = 100000; % a large number ~ infinity
    tinyVal = 10^(-10); % a small number ~ 0
    
    %% Initiallization
    d_matr = zeros(n+1); % control
    u = u_initiation(); % value function

    %% Main part
    change = hugeVal; % current difference between value functions in sequential iterations
    k = 0; % iteration number
    while (change > iter_tol)
        change=0;
        
        % alternating meshpoint orderings (“Fast Sweeping”)
        switch mod(k,4)
            case 0
                irange = 0:n;
                jrange = 0:n;
            case 1
                irange = 0:n;
                jrange = n:-1:0;
            case 2
                irange = n:-1:0;
                jrange = n:-1:0;
            case 3
                irange = n:-1:0;
                jrange = 0:n;
            otherwise
                error('weird k');
        end
        
        for i = irange
            for j = jrange
                
                if (i+j > n) % skip the half of the domain if x1+x2 > 1
                    d_matr(i+1,j+1)=NaN;
                    continue;
                end
                
                x1 = i*h;
                x2 = j*h;
                x = [x1, x2];
                if (x2 > fb) && (x2 < 1-fb) % skip fixed recovery and failure zones
                    
                    u_new = hugeVal;
                    for d = [0, d_max]
                        if norm(f(x, d))==0
                            continue;
                        end
                        tau = tau_func(x, d, i, j);
                        xtilde = x + tau * f(x, d); % new state
                        % value of u under control d
                        u_possible = tau * K(x, d) + u_interped(u, xtilde, i , j);
                        
                        if (u_possible < u_new)
                            u_new = u_possible;
                            d_new = d;
                        end
                    end
                    
                    %update the value function u at state x
                    if (u_new < u(i+1,j+1))
                        this_change = u(i+1,j+1) - u_new;
                        u(i+1,j+1) = u_new;
                        d_matr(i+1,j+1) = d_new;
                        if (this_change > change)
                            change = this_change;
                        end
                    end
                end
            end
        end
        k = k + 1;
        % print the current difference between value functions in sequential iterations
        change
    end
    
    
    
    %% Visualization of the optimal control and value function
    show_plots()
    
    
    %% Helping functions
    
    
    %% Initializaion of the value function u
    function u = u_initiation()
        u =  ones(n+1)*hugeVal;
        
        % skip the half of the domain where x1 + x2 > 1
        for ii = 0:n
            for jj =(n-ii+1):n
                u(ii+1,jj+1)=NaN;
            end
        end
         
        % value function = 0 at the recovery zone
        for ii = 0:n
            for jj = 0:(n-ii)
                if   (jj)*h < fb
                    u(ii+1,jj+1) = 0;
                end
            end
        end
 
    end

    %% Instantaneous cost 
    function y = K(~, d)
        y  = d + sigma;
    end
    
    %% Direction of movement at state x under control d
    function y = f(x, d)
        
        % transformation into (p, q) coordinates
        p = x(2);
        q = (1-x(1)-x(2))/(1-x(2));

        % direction of movement in (p, q) coordinates
        sum_p=0;
        for z=0:n_neigh
            sum_p = sum_p + p^z;
        end
        dq = q*(1-q)*(bv/(n_neigh+1)*sum_p-c);
        dp = p*(1-p)*(ba/(n_neigh+1) - q*(bv-c)-d);
        
        % transformation into (x_G, x_V, x_D) coordinates
        y = [-dq*(1-p) - dp*(1-q), dp];
   
    end
    
    
    %% Find time of movement tau
    function y=tau_func(x, d, i, j)
        
        func = f(x, d);
        
        assert(norm(func)>0);
        
        if (func(1) == 0)
            y = h / abs(func(2));
        elseif (func(2) == 0)
            y = h / abs(func(1));
        else
            if (func(1) * func(2) > 0)
                x1_int = [(i+sign(func(1))) * h, j * h];
                x2_int = [i * h, (j+sign(func(2))) * h];
            elseif (abs(func(2)) > abs(func(1)))
                x1_int = [(i+sign(func(1))) * h, (j + sign(func(2))) * h];
                x2_int = [i * h, (j+sign(func(2))) * h];
            else
                x1_int = [(i+sign(func(1))) * h, j * h];
                x2_int = [(i+sign(func(1))) * h, (j + sign(func(2))) * h];
            end
            
            k1 = x2_int(1) - x1_int(1);
            k2 = x1_int(2) - x2_int(2);
            kc = - (x1_int(2) * k1 + x1_int(1) * k2);
            y = - (kc + k1*x(2) + k2*x(1)) / (k1*func(2) + k2*func(1));
            
        end
        
        if (isnan(y) || isinf(y) || (y <= 0))
            error('Cannot compute Tau!');
        end
    end
    
    %% Return value funcion at state xtilde
    % u interped at (x + tau * f(x,b))
    function y =  u_interped(u, xtilde, i, j)
        
        dist = h*sqrt(2);
        
        % there are 6 possible combinations of 2 neighboring meshpoints.
        
        %%%%%% 3 %%%%%%%%%%%%%
        %%%%%--------%%%%%%%%%   *---->|
        %%%4 -        - 1 %%%%   ^     |
        %%%%%-          -%%%%%   |     \/
        %%%% 2 -        - 6 %%   |<----*
        %%%%%%%%%--------%%%%%
        %%%%%%%%%%%%% 5 %%%%%%
        
        %*----> is the direction where * is point that we include
        
        % value function at state xtilde is approximated by interpolation
        % using the neighboring meshpoint values.
          
            %1
        if (xtilde(1) >= i*h) && (xtilde(2) > j*h)
            x1_int = [i*h, (j+1)*h];
            gamma = norm(xtilde-x1_int) / dist;
            y = u(i+1, j+1+1)*(1-gamma) + u(i+1+1, j+1)*gamma;
            %2
        elseif (xtilde(1) <= i*h) && (xtilde(2) < j*h) && (i~=0)
            x1_int = [i*h, (j-1)*h];
            gamma = norm(xtilde-x1_int) / dist;
            y = u(i+1, j-1+1)*(1-gamma) + u(i-1+1, j+1)*gamma;    
            %3
        elseif (xtilde(1) ~= i*h) && (abs(xtilde(2) - (j+1)*h) < tinyVal)
            x1_int = [(i-1)*h, (j+1)*h];
            gamma = norm(xtilde-x1_int) / h;
            y = u(i-1+1, j+1+1)*(1-gamma) + u(i+1, j+1+1)*gamma;
            %4
        elseif (abs(xtilde(1) - (i-1)*h) < tinyVal) && (xtilde(2) ~= (j+1)*h)
            x1_int = [(i-1)*h, j*h];
            gamma = norm(xtilde-x1_int) / h;
            y = u(i-1+1, j+1)*(1-gamma) + u(i-1+1, j+1+1)*gamma;  
            %5
        elseif (xtilde(1) ~= i*h) && (abs(xtilde(2) - (j-1)*h) < tinyVal)
            x1_int = [(i+1)*h, (j-1)*h];
            gamma = norm(xtilde-x1_int) / h;
            y = u(i+1+1, j-1+1)*(1-gamma) + u(i+1, j-1+1)*gamma;
            %6
        elseif (abs(xtilde(1) - (i+1)*h) < tinyVal) && (xtilde(2) ~= (j-1)*h)
            x1_int = [(i+1)*h, j*h];
            gamma = norm(xtilde-x1_int) / h;
            y = u(i+1+1, j+1)*(1-gamma) + u(i+1+1, j-1+1)*gamma;
        elseif i==0 && (xtilde(2) < j*h)
            y = u(i+1, j-1+1);
        elseif i==0 && (xtilde(2) > j*h)
            y = u(i+1, j+1+1);
        else
            error('We are not in any quadrant at all!');
        end
        
    end
    
    
    
    
    %% Visualization of the optimal control and value function
    function show_plots()
        
        [X,Y] = meshgrid(0:h:1, 0:h:1);
        [X,Y] = transf(X',Y'); % transformation into a regular triangular mesh
        uu = u;
        for ii = 0:n
            for jj = 0:n
                if jj*h < fb  || jj*h > 1- fb
                    uu(ii+1, jj+1) = NaN;
                end
                if uu(ii+1, jj+1)>10
                    uu(ii+1, jj+1)=10;
                end
                
            end
        end

        figure
        mymap = [parula(2) ; 0, 1, 0];
        colormap(mymap)
        pcolor(X, Y, d_matr)% plot optimal control
        hold on
        contour(X, Y, uu, 'r')% plot value function
        axis equal
        axis([0 1 0 1])
        shading flat

    end
    
    
    %% Transformation to simplex x1 + x2 + x3 = 1
    function [X_tr,Y_tr]=transf(X,Y)

        T=[1 1/2; 0 sqrt(3)/2]; % linear transformation into a regular triangular mesh
        X_tr=X; Y_tr=Y;
        for i1 = 1:length(X(:,1))
            for i2 = 1:length(X(:,1))
                var=(T*[X(i1,i2), Y(i1,i2)]')';
                X_tr(i1,i2)=var(1);
                Y_tr(i1,i2)=var(2);
            end
        end
    end

    
end


%% References
% [1] Kaznatcheev A, Vander Velde R, Scott JG, Basanta D.
% 2017 Cancer treatment scheduling and dynamic
% heterogeneity in social dilemmas of tumour acidity
% and vasculature. Br. J. Cancer 116, 785–792.




