% Topic 1: Asset Returns

clear all;

% nvda
NVDA=getMarketDataViaYahoo('NVDA','1999-02-01',datetime('today'),'1d');
dates=NVDA.Date;
nvda=NVDA.Close;
% getMarketDataViaYahoo is a matlab function which extracts a table of information from the Yahoo website. The downloaded variables are Date, Open, High, Low, Close, AdjClose, and Volume.


% Create returns 
Rnvda = 100*(nvda(2:end) -nvda(1:end-1)) ./ nvda(1:end-1);
% Create log returns
rnvda = 100*(log(nvda(2:end)) - log(nvda(1:end-1)));
% Adjust date vector to match returns
dates2 = dates(2:end,:);

% S&P 500
SP500 = getMarketDataViaYahoo('^GSPC','1999-02-01',datetime('today'),'1d');
sp500 = SP500.Close;
rsp500 = 100*(log(sp500(2:end)) - log(sp500(1:end-1)));

% Plot Price of nvda Stock
figure(1);
plot(dates,nvda);
ylabel('Dollars');
title('Price of NVIDIA Stock');
print('Topic1_1','-dpdf');
box off

% Plot returns
figure(2);
plot(dates2,Rnvda);
ylabel('Percentage Return');
title('Simple Net Return on NVIDIA Stock');
print('Topic1_2','-dpdf');
box off

% Plot log returns
figure(3);
plot(dates2,rnvda);
ylabel('Percentage Return');
title('Log Return on nvda Stock');
print('Topic1_3','-dpdf');
box off


% Calculate moments of log returns
m1 = mean(rnvda);
s1 = std(rnvda);
sk1 = skewness(rnvda);
kr1 = kurtosis(rnvda);

fprintf('\n');
fprintf('NVDA\n');
fprintf('Mean    %.3f\n' , m1);
fprintf('StdDev  %.3f\n' , s1);
fprintf('Skew    %.3f\n' , sk1);
fprintf('Kurt    %.3f\n' , kr1);
fprintf('\n');

% Plot level of S&P Series
figure(4);
plot(dates,sp500);
ylabel('Dollars');
title('Price of S&P Stock Index');
print('Topic1_4','-dpdf');
box off

% Plot log return on S&P
figure(5);
plot(dates2,rsp500);
ylabel('Percentage Return');
title('Simple Net Return on S&P Stock Index');
print('Topic1_5','-dpdf');
box off

% Calculate moments of log returns
m1 = mean(rnvda);
s1 = std(rnvda);
sk1 = skewness(rnvda);
kr1 = kurtosis(rnvda);
m2 = mean(rsp500);
s2 = std(rsp500);
sk2 = skewness(rsp500);
kr2 = kurtosis(rsp500);

fprintf('\n');
fprintf('NVIDIA\n');
fprintf('Mean    %.3f\n' , m1);
fprintf('StdDev  %.3f\n' , s1);
fprintf('Skew    %.3f\n' , sk1);
fprintf('Kurt    %.3f\n' , kr1);
fprintf('\n');
fprintf('S&P 500\n');
fprintf('Mean    %.3f\n' , m2);
fprintf('StdDev  %.3f\n' , s2);
fprintf('Skew    %.3f\n' , sk2);
fprintf('Kurt    %.3f\n' , kr2);
fprintf('\n');

% Plot histogram on log returns
figure(6)
hist(rnvda,100)
title('Log Return on NVIDIA Stock');
print('Topic1_6','-dpdf');
box off

% Reprint using narrower X axis
figure(7)
hist(rnvda,100)
xlim([-8 8])
title('Log Return on NVIDIA Stock');
print('Topic1_7','-dpdf');
box off

% Plot nonparametric estimate of density
figure(8)
[f,x] = ksdensity(rnvda,(-6:.1:6));
g = normpdf((x-m1)/s1)/s1;
plot(x,f,x,g,'--')
title('Log Return on NVIDIA Stock');
legend('NonParametric','Normal','Location','northeast');
print('Topic1_8','-dpdf');
box off

% Plot nonparametric estimate of density of S&P log returns
figure(9)
[f,x] = ksdensity(rsp500,(-4:.1:4));
g = normpdf((x-m2)/s2)/s2;
plot(x,f,x,g,'--')
title('Log Return on S&P 500');
legend('NonParametric','Normal','Location','northeast');
print('Topic1_9','-dpdf');
box off

% Plot autocorrelation functions
figure(10)
autocorr(rnvda)
title('Log Return on NVIDIA Stock');
print('Topic1_10','-dpdf');
box off

figure(11)
autocorr(rsp500)
title('Log Return on S&P 500');
print('Topic1_11','-dpdf');
box off

% Plot autocorrelation functions of absolute returns
figure(12)
autocorr(abs(rnvda))
title('Absolute Log Return on NVIDIA Stock');
print('Topic1_12','-dpdf');
box off

figure(13)
autocorr(abs(rsp500))
title('Absolute Log Return on S&P 500');
print('Topic1_13','-dpdf');
box off


