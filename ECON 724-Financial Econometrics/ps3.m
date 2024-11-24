% Topic 4: ARCH Models

clear all

% Nike
Nike = getMarketDataViaYahoo('NKE','1-Jan-1985','31-Dec-2021','1d');
P = Nike.Close;
dates = Nike.Date;
dates2 = dates(2:end,:);
r = 100*(P(2:end) - P(1:end-1))./P(1:end-1);

%Q2
figure(1);
plot(dates,P);
ylabel('Dollars')
title('Price of Nike Stock');
print('Topic4_1','-dpdf');
box off

figure(2);
plot(dates2,r);
ylabel('Percentage Return')
title('Return on Nike Stock');
print('Topic4_2','-dpdf');
box off

% Q3
% estimate AR(1) model
p = 1; % autoregressive order
Y = r(p+1:end); %truncate returns for initial condition(s)
r_lag = lagmatrix(r,1:p); Y_lag = r_lag(p+1:end,:); %matrix of lagged returns
%estimate model & extract coefficients
[V,se,beta] = hac(Y_lag,Y,'type','HC','weights','HC3','display','full'); 
% compute test statistic and p-value
W = beta(2:end)'*inv(V(2:end,2:end))*beta(2:end);
pval = 1-chi2cdf(W,size(beta,1)-1);
%display results
fprintf('Testing EMH with AR(%d) model\n',p);
fprintf('Wald %.3f\n' , W);
fprintf('pvalue %.3f\n' , pval);

%Q4
p = 5; % autoregressive order
Y = r(p+1:end); %truncate returns for initial condition(s)
r_lag = lagmatrix(r,1:p); Y_lag = r_lag(p+1:end,:); %matrix of lagged returns
%estimate model & extract coefficients
[V,se,beta] = hac(Y_lag,Y,'type','HC','weights','HC3','display','full'); 
% compute test statistic and p-value
W = beta(2:end)'*inv(V(2:end,2:end))*beta(2:end);
pval = 1-chi2cdf(W,size(beta,1)-1);
%display results
fprintf('Testing EMH with AR(%d) model\n',p);
fprintf('Wald %.3f\n' , W);
fprintf('pvalue %.3f\n' , pval);

%Q5
% estimate AR(5) model with lagged factors and their squares
p = 5; % autoregressive order
Y = r(p+1:end); % truncate returns for initial condition(s)
r_lag = lagmatrix(r,1:p); 
Y_lag = r_lag(p+1:end,:); % matrix of lagged returns
% Add squared terms for lagged factors
for i = 1:p
    Y_lag = [Y_lag r_lag(p+1:end,i).^2];
end
[V,se,beta] = hac(Y_lag,Y,'type','HC','weights','HC3','display','full'); % estimate model & extract coefficients
% compute test statistic and p-value
W = beta(2:end)'*inv(V(2:end,2:end))*beta(2:end);
pval = 1 - chi2cdf(W,size(beta,1)-1);
% display results
fprintf('Testing EMH with AR(%d)-squrare model including squared terms\n',p);
fprintf('Wald %.3f\n' , W);
fprintf('pvalue %.3f\n' , pval);
