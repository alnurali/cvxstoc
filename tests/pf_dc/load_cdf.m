function [Y, s, v, slack, pv, pq, base_mva] = load_cdf(filename)

% load (simplified) IEEE test case
fid = fopen(filename, 'r');
line = fgetl(fid);
base_mva = str2num(line(32:37));

line = fgetl(fid);
if (strcmp(line(1:16),'BUS DATA FOLLOWS'))
  % load bus data (bus types and initial voltagse)
  i = 1;
  pv = []; pq = []; slack = [];
  
  line = fgetl(fid);
  while (~strcmp(line(1:4), '-999'))
    bus(str2num(line(1:4))) = i;
    
    % read initial voltage
    vhat = str2num(line(28:33));
    theta = str2num(line(34:40))*(pi/180);
    v(i,1) = vhat * exp(j*theta);
    
    % read angle
    type = str2num(line(25:26));
    if (type == 0 | type == 1)
      pq = [pq; i];
    elseif (type == 2 | (type == 3 & ~isempty(slack)))
      pv = [pv; i];
    else
      slack = i;
    end
    
    % read power
    s(i,1) = str2num(line(60:67)) - str2num(line(41:49));
    s(i,1) = s(i,1) + j*(str2num(line(68:75)) - str2num(line(50:59)));
    s(i,1) = s(i,1) / base_mva;
    
    line = fgetl(fid);
    i = i + 1;
  end
  n = i-1;
else
  disp('No bus data found');
end


line = fgetl(fid);
if (strcmp(line(1:19),'BRANCH DATA FOLLOWS'))
  % load branch data
  Y = sparse(n,n);
  
  line = fgetl(fid);
  while (~strcmp(line(1:4), '-999'))
    i = bus(str2num(line(1:4)));
    k = bus(str2num(line(6:9)));
    r = str2num(line(20:29));
    x = str2num(line(30:40));
    Y(i,k) = -1/(r + j*x);
    line = fgetl(fid);
  end
  Y = Y + Y.';
  Y = Y - spdiags(sum(Y,2), 0, n, n);
else
  disp('No branch data found');
end
