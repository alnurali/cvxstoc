clear all; fclose all; close all; clc; rand('seed',1);

out_fn = 'pf_dc.mat';


%% form B matrix
%{
note that:

Z := impedance matrix = resistance + j * reactance, j := sqrt(-1)

Y := admittance matrix = inv(Z)

    as it turns out:

    for a neq b, bus a connected to bus b:
        Y_ab = -1 / (r_ab + j * x_ab)
        where
        r_ab := resistance between bus a and bus b
        x_ab := reactance between bus a and bus b

    for a neq b, bus a not connected to bus b:
        Y_ab = 0

    for a = b:
        Y_aa = -sum_b Y_ab (i.e. minus the sum along the a'th row)

B := imag(Y), after first zeroing out the resistances in Y
%}
[Y,s,v,slack,pv,pq,mva] = load_cdf('ieee14cdf.txt');
n = size(Y,1);

B = 1 ./ Y;
B = imag(B);
B = -1 ./ B;
B(find(B == -inf)) = 0;

B_nodiag = B - diag(diag(B));
B = B_nodiag + diag(sum(B_nodiag,2));


%% form bus indices for generators and loads
gen_idxes = [1;2]; % for this problem, we'll just take nodes 1 (slack) and 2 (pv gen) as generators...
wind_idxes = 13; % ...and take node 13 as the wind generator

load_idxes = setdiff( [1:n], union(gen_idxes,wind_idxes) );

num_gens = length(gen_idxes);
num_winds = length(wind_idxes);
num_loads = length(load_idxes);


%% form initial real power conditions
p = real(s); % real powers (for the loads)

p_w1 = rand;
p_w2 = rand;

l = zeros(num_gens,1); % generator lower bounds (i.e., generators can't consume power)
u = abs(sum(p(load_idxes)))/num_gens * ones(num_gens,1); % generator upper bounds (i.e., no generator can shoulder more than half the (total) load)

%{
note that:
v is expressed as a + j * b

which implies that:
voltage angle := theta = arctan(b/a)
voltage magnitude := vhat = sqrt(a^2 + b^2)
%}
theta = atan( imag(v) ./ real(v) ); % voltage angles for generators (and, actually for the loads, which we will not use, however)


%% form costs
C_g = [2 0; 0 4]; % quadratic cost coefficients for producing power (not used)
c_g = [5; 2]; % linear cost coefficients for producing power

C_w = -5; % quadratic (negative) revenue coefficient for selling power (not used)
c_w = -2; % linear (negative) revenue coefficient for selling power


%% form adjacency matrix
A = [];
edge_ctr = 0;
for i=1:n-1
    for j=i+1:n
        if( B(i,j) ~= 0 )
            edge_ctr = edge_ctr+1;
            A(i,edge_ctr) = +1; % edge directionality is abtirary, the solution (i.e., p_g and p_lines) will be rescaled accordingly
            A(j,edge_ctr) = -1;
        end
    end
end
m = size(A,2);


%% write to disk
save(out_fn, 'B');
save(out_fn, 'A');

save(out_fn, 'gen_idxes', '-append');
save(out_fn, 'wind_idxes', '-append');
save(out_fn, 'load_idxes', '-append');

save(out_fn, 'p', '-append');

save(out_fn, 'l', '-append');
save(out_fn, 'u', '-append');

save(out_fn, 'theta', '-append');

save(out_fn, 'C_g', '-append');
save(out_fn, 'c_g', '-append');

save(out_fn, 'C_w', '-append');
save(out_fn, 'c_w', '-append');


%% solve the deterministic equivalent of a two-stage DC optimal power flow problem
cvx_begin

    % first stage variables
    variable p_g1(1);
    
    % second stage variables        
    variable p_g2_w1(1);
    variable p_g2_w2(1);
    
    variable z_w1(1);
    variable z_w2(1)    
    
    variable p_lines_w1(m);
    variable p_lines_w2(m);
    
    minimize( (1/2)*( c_g'*[p_g1; p_g2_w1] + c_w'*z_w1 ) + (1/2)*( c_g'*[p_g1; p_g2_w2] + c_w'*z_w2 ) ); % this cost function is purely second stage
    
    subject to
    
        % first stage constraints
        0 <= p_g1 <= 1;
        
        % second stage constraints
        A*p_lines_w1 == [p_g1; p_g2_w1; p(load_idxes(1:end-1)); p_w1-z_w1; p(load_idxes(end))];
        A*p_lines_w2 == [p_g1; p_g2_w2; p(load_idxes(1:end-1)); p_w2-z_w2; p(load_idxes(end))];
                        
        0 <= p_g2_w1 <= 1;
        0 <= p_g2_w2 <= 1;
        
        0 <= z_w1 <= p_w1;
        0 <= z_w2 <= p_w2;
        
        abs(p_lines_w1) <= 1;
        abs(p_lines_w2) <= 1;        

cvx_end

fprintf( 'power supplied by generator 1 = [%f].\n', p_g1 )

avg_wind = (p_w1+p_w2)/2;
fprintf( '\npower supplied by wind generator, scenario 1 = [%f]    scenario 2 = [%f]    avg. = [%f].\n', p_w1, p_w2, avg_wind )

fprintf( 'wind power sold, scenario 1 = [%f]    scenario 2 = [%f].\n', z_w1, z_w2 )

avg_g2 = (p_g2_w1 + p_g2_w2)/2;
fprintf( '\npower supplied by generator 2, scenario 1 = [%f]    generator 2, scenario 2 = [%f]    avg. = [%f].\n', p_g2_w1, p_g2_w2, avg_g2 )

fprintf( '\ntotal power supplied, scenario 1 = [%f]    scenario 2 = [%f].\n', p_g1+p_g2_w1+p_w1, p_g1+p_g2_w2+p_w2 )

demand = sum(p(load_idxes));
fprintf( 'power demanded = [%f].\n', demand )


%% plot network
B_binary = B ~= 0;
coords = [ 0        2;
           1        1;
           3        0.25;
           2.85     1.5;
           2        1.75;
           2        2;
           3        1.75;
           3.25     1.85;
           2.85     2;
           2.65     2.25;
           2.45     2.35;
           1.75     2.55
           2        2.8
           2.5      2.65];
       
f = figure;
hold on;

% draw the edges between vertices
gplot(B_binary, coords, '-k');

% draw symbols and power generation/consumption @ each vertex
mu = 0;
sigma = 1;
lognorm_mean = exp(mu+sigma^2/2);

shift_x = 0;
shift_y = 0.125;
for i=1:n    
    if(~isempty(find(gen_idxes==i))) % generators
        plot( coords(i,1), coords(i,2), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 12 );
        if(i==1)
            text( coords(i,1)+shift_x, coords(i,2)+shift_y, num2str(p_g1), 'FontSize', 10 );
        else
            text( coords(i,1)+shift_x, coords(i,2)+shift_y, 'sec. stg.', 'FontSize', 10 );
        end
    elseif(~isempty(find(wind_idxes==i))) % wind generators
        plot( coords(i,1), coords(i,2), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 12 );
        text( coords(i,1)+shift_x, coords(i,2)+shift_y, num2str(lognorm_mean), 'FontSize', 10 );
    else % loads
        plot( coords(i,1), coords(i,2), '.k', 'MarkerSize', 15 );           
        text( coords(i,1)+shift_x, coords(i,2)+shift_y, num2str(p(i)), 'FontSize', 10 );
    end    
end

axis off;
hold off;
print( f, '-dpdf', 'grid.pdf' );

save(out_fn, 'B_binary', '-append');
save(out_fn, 'coords', '-append');