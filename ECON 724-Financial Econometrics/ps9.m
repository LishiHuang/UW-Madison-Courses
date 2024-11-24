% Topic 8: Factor Pricing

clear all

% Q1
% Define the list of stock symbols
symbols = {'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON'};

% Initialize empty arrays for prices and returns
P = [];
R = [];

% Loop through each symbol and retrieve market data
for i = 1:length(symbols)
    data = getMarketDataViaYahoo(symbols{i}, '1-Dec-2001', '31-Dec-2021', '1d');
    P = [P, data.Close];
    R = [R, 100*(data.Close(2:end) - data.Close(1:end-1)) ./ data.Close(1:end-1)];
end

% Calculate portfolio weights using quadratic programming
n = size(R, 2);
V = cov(R);
N = ones(n, 1);
w = quadprog(V, [], [], [], N', 1);

% Calculate portfolio returns and standard deviation
Rw = R * w;
s = std(Rw);
mu = mean(Rw);

% Display results
fprintf('Weights:\n');
disp(w);
fprintf('Mean of portfolio: %.3f\n', mu);
fprintf('Standard deviation of portfolio: %.3f\n', s);
fprintf('\n');

% Identify stocks with negative weights (short positions)
short_positions = symbols(w < 0);
fprintf('Stocks with short positions: %s\n', strjoin(short_positions, ', '));

Z = zeros(n,1);
w2 = quadprog(V, [], [], [], N', 1,Z);

Rw2 = R * w2;
s2 = std(Rw2);
mu2 = mean(Rw2);

fprintf('Weights:\n');
disp(w2);
fprintf('Mean of portfolio: %.3f\n', mu2);
fprintf('Standard deviation of portfolio: %.3f\n', s2);
fprintf('\n');


% MLE factor analysis with 2 factors
[Bf, sig, T, stats, FF] = factoran(R, 2, 'scores', 'regression');

% Construct covariance matrix of returns
Cov = Bf * Bf' + diag(sig);

% Estimate unrestricted minimum variance portfolio weights
w_unrestricted = inv(V) * ones(n, 1) / (ones(n, 1)' * inv(V) * ones(n, 1));

% If weights include short selling, re-estimate weights under no-short-selling restriction
if any(w_unrestricted < 0)
    Aeq = ones(1, n);
    beq = 1;
    lb = zeros(n, 1);
    ub = inf(n, 1);
    w_no_short_selling = quadprog(V, [], [], [], Aeq, beq, lb, ub);
else
    w_no_short_selling = w_unrestricted;
end

% Display results
fprintf('Unrestricted Minimum Variance Portfolio Weights:\n');
disp(w_unrestricted);
fprintf('\n');
fprintf('No-Short-Selling Minimum Variance Portfolio Weights:\n');
disp(w_no_short_selling);
fprintf('\n');
