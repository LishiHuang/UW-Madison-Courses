% PS 4

clear all
%Q1

% Parameters
alpha = 0.2;
w = 0.8;
T = 9000;

% Simulate ARCH(1) series
e = zeros(T, 1);
sigma_sq = zeros(T, 1);
sigma_sq(1) = w / (1 - alpha); % initial variance
for t = 2:T
    sigma_sq(t) = w + alpha * e(t-1)^2;
    e(t) = sqrt(sigma_sq(t)) * randn; % generate Gaussian white noise
end

% Calculate moments
mean_e = mean(e);
var_e = var(e);
skewness_e = skewness(e);
kurtosis_e = kurtosis(e);

% Display results
fprintf('Mean of e: %.4f\n', mean_e);
fprintf('Variance of e: %.4f\n', var_e);
fprintf('Skewness of e: %.4f\n', skewness_e);
fprintf('Kurtosis of e: %.4f\n', kurtosis_e);


% Estimate parameters using Maximum Likelihood Estimation
Q1 = garch('ARCHLags',1,'Offset',NaN);% Define the ARCH(1) model
estQ1 = estimate(Q1,e); % Estimate the model parameters
v1 = sqrt(infer(estQ1,e));

% Parameters
alpha_2 = 0.5;
w_2 = 0.5;
T_2 = 9000;

% Simulate ARCH(1) series
e_2 = zeros(T_2, 1);
sigma_sq_2 = zeros(T_2, 1);
sigma_sq_2(1) = w_2 / (1 - alpha_2); % initial variance
for t = 2:T_2
    sigma_sq_2(t) = w_2 + alpha_2 * e_2(t-1)^2;
    e_2(t) = sqrt(sigma_sq_2(t)) * randn; % generate Gaussian white noise
end

% Calculate moments
mean_e_2 = mean(e_2);
var_e_2 = var(e_2);
skewness_e_2 = skewness(e_2);
kurtosis_e_2 = kurtosis(e_2);

% Display results
fprintf('Mean of e_2: %.4f\n', mean_e_2);
fprintf('Variance of e_2: %.4f\n', var_e_2);
fprintf('Skewness of e_2: %.4f\n', skewness_e_2);
fprintf('Kurtosis of e_2: %.4f\n', kurtosis_e_2);

% Estimate parameters using Maximum Likelihood Estimation
Q2 = garch('ARCHLags',1,'Offset',NaN);% Define the ARCH(1) model
estQ2 = estimate(Q2,e_2); % Estimate the model parameters
v2 = sqrt(infer(estQ2,e_2));

%Q3
Ford = getMarketDataViaYahoo('F','1-Jan-1985','31-Dec-2021','1d');
P = Ford.Close;
dates = Ford.Date;
dates2 = dates(2:end,:);
r = 100*(log(P(2:end)) - log(P(1:end-1)));

% ARCH(1) Model for Ford

ARCH1 = garch('ARCHLags',1,'Offset',NaN);
estARCH1 = estimate(ARCH1,r);
vv1 = sqrt(infer(estARCH1,r));

figure(1);
plot(dates2,vv1);
ylim([0 15]);
ylabel('Percentage Return');
title('Estimated Standard Deviation using ARCH(1)');
print('Topic4_6','-dpdf');
box off

% ARCH(2) Model 

ARCH2 = garch('ARCHLags',1:2,'Offset',NaN);
estARCH2 = estimate(ARCH2,r);
vv2 = sqrt(infer(estARCH2,r));

figure(2);
plot(dates2,vv2);
ylim([0 15]);
ylabel('Percentage Return');
title('Estimated Standard Deviation using ARCH(10)');
print('Topic4_7','-dpdf');
box off

% ARCH(10) Model 
ARCH10 = garch('ARCHLags',1:10,'Offset',NaN);
estARCH10 = estimate(ARCH10,r);
vv10 = sqrt(infer(estARCH10,r));
figure(3);
plot(dates2,vv10);
ylim([0 15]);
ylabel('Percentage Return');
title('Estimated Standard Deviation using ARCH(10)');
print('Topic4_8','-dpdf');
box off
