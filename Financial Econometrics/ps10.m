clear;
clc;

current_return = 2;
current_variance = 4;

current_conditional_VaR = -norminv(0.01, current_return, sqrt(current_variance));
fprintf('Current conditional VaR: %.3f\n', current_conditional_VaR);

updated_conditional_VaR = -norminv(0.01, current_return, sqrt(current_variance + current_conditional_VaR^2));
fprintf('Updated conditional VaR: %.3f\n', updated_conditional_VaR);

start = '01-Dec-2001'; %start date for selected stocks
T = '31-Dec-2021'; %end date for selected stocks
freq = '1d'; %frequency
data = getMarketDataViaYahoo('NKE',start,T,freq); % price data
P = data.Close; % closing prices
R = 100*(P(2:end) - P(1:end-1)) ./ P(1:end-1); % net returns
VaR = -quantile(R,0.01); % 1% VaR
fprintf('1-day unconditional VaR: %.3f\n',VaR);

mu = mean(R); s = std(R);
VaR1 = -norminv(0.05,mu,s);
VaR2 = -quantile(R,0.05);
fprintf('Parametric:      %.3f\n',VaR1);
fprintf('Nonparametric:   %.3f\n',VaR2);