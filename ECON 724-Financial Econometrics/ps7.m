clear all;

P1 = getMarketDataViaYahoo('^GSPC','1-Jan-1995','31-Dec-2023','1d'); % GSPC
P2 = getMarketDataViaYahoo('^RUT','1-Jan-1995','31-Dec-2023','1d'); % RUT
dates = P1.Date;
R1 = 100*(P1.Close(2:end) - P1.Close(1:end-1)) ./ P1.Close(1:end-1);
R2 = 100*(P2.Close(2:end) - P2.Close(1:end-1)) ./ P2.Close(1:end-1);
dates2 = dates(2:end,:);

mu1 = mean(R1);
mu2 = mean(R2);
sd1 = std(R1);
sd2 = std(R2);
V = cov([R1,R2]);
rho = corr(R1,R2);
sig1 = sd1.^2;
sig2 = sd2.^2;
sig12 = V(2,1);
w1 = (sig1-sig12)/(sig1+sig2-2*sig12);
s1 = sqrt(sig1*((1-w1)^2) +sig2*(w1^2)+2*sig12*w1*(1-w1));
m1 = (1-w1)*mu1 + w1*mu2;

fprintf('\n');
fprintf('GSPC & RUT portfolio\n');
fprintf('mean GSPC return  	   %.3f\n' , mu1);
fprintf('mean RUT return  	   %.3f\n' , mu2);
fprintf('SD GSPC return  	   %.3f\n' , sd1);
fprintf('SD RUT return  	   %.3f\n' , sd2);
fprintf('Correlation	  	   %.3f\n' , rho);
fprintf('Cov Matrix	  	  \n') ;
disp(V);
fprintf('Weight on RUT  	   %.3f\n' , w1);
fprintf('Weight on GSPC  	   %.3f\n' , 1-w1);
fprintf('SD of portfolio  	   %.3f\n' , s1);
fprintf('\n');

%%

% Given variables
r_daily = 0.02; % daily risk-free rate
mu_daily = [mu1; mu2]; % daily mean returns
sigma_daily = V; % daily covariance matrix

% Transformation to annual
r_annual = 252 * r_daily;
mu_annual = 252 * mu_daily;
sigma_annual = 252^2 * sigma_daily;

% (a) Calculate the weight vector w for the minimum variance portfolio
inv_sigma_annual = inv(sigma_annual);
ones_vector = ones(length(mu_annual), 1);
w_minvar = inv_sigma_annual * ones_vector / (ones_vector' * inv_sigma_annual * ones_vector);

% Display w for minimum variance portfolio
disp('Weight vector for minimum variance portfolio:');
disp(w_minvar');

% (b) Calculate the weight vector w for the tangency portfolio
risk_aversion = 2; % example value for risk aversion coefficient
w_tangency = (inv(sigma_annual) * (mu_annual - r_annual * ones_vector)) / (ones_vector' * inv(sigma_annual) * (mu_annual - r_annual * ones_vector));

% Display w for tangency portfolio
disp('Weight vector for tangency portfolio:');
disp(w_tangency');

%%

clear all;

% Get daily stock price data for the specified tickers and date range
AMD = getMarketDataViaYahoo('AMD','1-Jan-1985','31-Dec-2023','1d'); % AMD
GLW = getMarketDataViaYahoo('GLW','1-Jan-1985','31-Dec-2023','1d'); % GLW
HUM = getMarketDataViaYahoo('HUM','1-Jan-1985','31-Dec-2023','1d'); % HUM
WHR = getMarketDataViaYahoo('WHR','1-Jan-1985','31-Dec-2023','1d'); % WHR

% Calculate log returns
R_AMD = 100 * log(AMD.Close(2:end) ./ AMD.Close(1:end-1));
R_GLW = 100 * log(GLW.Close(2:end) ./ GLW.Close(1:end-1));
R_HUM = 100 * log(HUM.Close(2:end) ./ HUM.Close(1:end-1));
R_WHR = 100 * log(WHR.Close(2:end) ./ WHR.Close(1:end-1));

% Concatenate the returns into a matrix
returns_matrix = [R_AMD, R_GLW, R_HUM, R_WHR];

% Estimate mean returns and the covariance matrix for the sample period
mu = mean(returns_matrix)';
cov_matrix = cov(returns_matrix);

% Set up optimization problem
n = length(mu);
Aeq = ones(1, n);
beq = 1;
lb = zeros(n, 1);
ub = ones(n, 1);
options = optimoptions('fmincon', 'Display', 'off');

% (b) Calculate weights for the minimum variance portfolio
w_minvar = fmincon(@(w) w' * cov_matrix * w, zeros(n, 1), [], [], Aeq, beq, lb, ub, [], options);
mu_minvar = mu' * w_minvar;
std_minvar = sqrt(w_minvar' * cov_matrix * w_minvar);

% Display results for minimum variance portfolio
disp('Weights for Minimum Variance Portfolio:');
disp(w_minvar');
disp('Expected Return of Minimum Variance Portfolio:');
disp(mu_minvar);
disp('Portfolio Standard Deviation for Minimum Variance Portfolio:');
disp(std_minvar);

% (c) Efficient Frontier calculation
num_portfolios = 10000;
weights = rand(num_portfolios, 4);
weights = weights ./ sum(weights, 2); 
portfolio_returns = weights * mu;
portfolio_std_devs = sqrt(diag(weights * (cov_matrix * weights')));
figure;
scatter(portfolio_std_devs, portfolio_returns, 10, 'filled');
hold on;
xlabel('STD');
ylabel('EXPECTED RETURN');
title('EFFICIENT FROUNTIER');
grid on;

% (d) Calculate the tangency portfolio
r_annual = 0.02; % annual risk-free rate
r_daily = 0.000722; % daily risk-free rate
mu_annual = 252 * mu; % annual mean returns
cov_matrix_annual = 252^2 * cov_matrix; % annual covariance matrix
w_tangency = (inv(cov_matrix_annual) * (mu_annual - r_annual * ones(n, 1))) / ...
    (ones(1, n) * inv(cov_matrix_annual) * (mu_annual - r_annual * ones(n, 1)));
mu_tangency = mu_annual' * w_tangency;
std_tangency = sqrt(w_tangency' * cov_matrix_annual * w_tangency);

% Display results for tangency portfolio
disp('Weights for Tangency Portfolio:');
disp(w_tangency');
disp('Expected Return of Tangency Portfolio:');
disp(mu_tangency);
disp('Portfolio Standard Deviation for Tangency Portfolio:');
disp(std_tangency);

%%

PM=getMarketDataViaYahoo('^GSPC','1-Jan-2004','31-Dec-2023','1mo'); % GSPC
AMDm = getMarketDataViaYahoo('AMD','1-Jan-2004','31-Dec-2023','1mo'); % AMD
GLWm = getMarketDataViaYahoo('GLW','1-Jan-2004','31-Dec-2023','1mo'); % GLW
HUMm = getMarketDataViaYahoo('HUM','1-Jan-2004','31-Dec-2023','1mo'); % HUM
WHRm = getMarketDataViaYahoo('WHR','1-Jan-2004','31-Dec-2023','1mo'); % WHR

RM = 100*(PM.Close(2:end) - PM.Close(1:end-1)) ./ PM.Close(1:end-1);
R_AMD = 100 * log(AMDm.Close(2:end) ./ AMDm.Close(1:end-1));
R_GLW = 100 * log(GLWm.Close(2:end) ./ GLWm.Close(1:end-1));
R_HUM = 100 * log(HUMm.Close(2:end) ./ HUMm.Close(1:end-1));
R_WHR = 100 * log(WHRm.Close(2:end) ./ WHRm.Close(1:end-1));

d = fetch(fred,'GS1M','1-Jan-2004','31-Dec-2023'); % download one-month TBill
D = d.Data;
Rf = D(2:end,2); % select interest rate (column) and subset to match dates of series (row)
Rf = Rf/12; % scale as monthly percentage

% Create excess returns
Rm_s = RM - Rf; % S&P 500 (market)
R_AMDS = R_AMD - Rf; 
R_GLWS = R_GLW - Rf; 
R_HUMS = R_HUM - Rf; 
R_WHRS = R_WHR - Rf;

% estimate market regression/capital line for each firm and report output
fprintf('----------- Market Regressions -----------\n');
fprintf('AMD:\n');
hac(Rm_s,R_AMDS,'type','HC','weights','HC3','display','full');
fprintf('\n');

fprintf('Corning:\n');
hac(Rm_s,R_GLWS,'type','HC','weights','HC3','display','full');
fprintf('\n');

fprintf('Humana:\n');
hac(Rm_s,R_HUMS,'type','HC','weights','HC3','display','full');
fprintf('\n');
fprintf('----------------------------------------\n');

fprintf('Whirlpool:\n');
hac(Rm_s,R_WHRS,'type','HC','weights','HC3','display','full');
fprintf('\n');
fprintf('----------------------------------------\n');

% Define the sub-sample periods
subsample_periods = {'2004-2008', '2009-2013', '2014-2018', '2019-2023'};
subsample_start_dates = {'1-Jan-2004', '1-Jan-2009', '1-Jan-2014', '1-Jan-2019'};
subsample_end_dates = {'31-Dec-2008', '31-Dec-2013', '31-Dec-2018', '31-Dec-2023'};

% Loop over each sub-sample period
for i = 1:numel(subsample_periods)
    % Extract data for the current sub-sample period
    start_date = subsample_start_dates{i};
    end_date = subsample_end_dates{i};
    
    % Filter data for the sub-sample period
    P_subsample=getMarketDataViaYahoo('^GSPC',start_date,end_date,'1mo'); % GSPC
    RP_subsample=100 * log(P_subsample.Close(2:end) ./ P_subsample.Close(1:end-1));
    WHR_subsample = getMarketDataViaYahoo('WHR', start_date, end_date, '1mo');
    R_WHR_subsample = 100 * log(WHR_subsample.Close(2:end) ./ WHR_subsample.Close(1:end-1));
    
    % Fetch risk-free rate data for the sub-sample period
    d_subsample = fetch(fred, 'GS1M', start_date, end_date);
    D_subsample = d_subsample.Data;
    Rf_subsample = D_subsample(2:end, 2) / 12;
    
    % Calculate excess returns for the sub-sample period
    Rm_subsample = RP_subsample - Rf_subsample; 
    R_WHR_subsample_s = R_WHR_subsample - Rf_subsample;
    
    % Estimate market model regression for the sub-sample period
    fprintf('Market Model Regression for Whirlpool (%s):\n', subsample_periods{i});
    hac(Rm_subsample, R_WHR_subsample_s, 'type', 'HC', 'weights', 'HC3', 'display', 'full');
    fprintf('\n');
end
