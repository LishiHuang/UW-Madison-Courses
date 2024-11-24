% Topic 8: Factor Pricing

clear all

% Stock prices and returns, monthly
P1 = getMarketDataViaYahoo('AAPL','1-Dec-2001','31-Dec-2021','1mo'); % Pfizer
P2 = getMarketDataViaYahoo('NVDA','1-Dec-2001','31-Dec-2021','1mo'); % Disney

R1 = 100*(P1.Close(2:end) - P1.Close(1:end-1)) ./ P1.Close(1:end-1);
R2 = 100*(P2.Close(2:end) - P2.Close(1:end-1)) ./ P2.Close(1:end-1);

% Fama-French Factors
% Extract data in columns B through E, rows 911 through 1150
T = readtable('F-F_Research_Data_Factors.CSV','Range','B911:E1150','ReadVariableNames',false);
RM = T.Var1;
SMB = T.Var2;
HML = T.Var3;
Rf = T.Var4;

% Create excess returns
R1s = R1 - Rf;
R2s = R2 - Rf;

% FF Regressions
fprintf('Full Sample Regressions: 2002-2021\n');

fprintf('Apple Market Regression\n');
hac([SMB,HML,RM],R1s,'type','HC','weights','HC3','display','full');

fprintf('Navida Market Regression\n');
hac([SMB,HML,RM],R2s,'type','HC','weights','HC3','display','full');

%% 

% Factor Pricing

clear all

% Stock prices and returns
P1 = getMarketDataViaYahoo('AXP','1-Dec-2001','31-Dec-2021','1mo'); 
P2 = getMarketDataViaYahoo('AMGN','1-Dec-2001','31-Dec-2021','1mo'); 
P3 = getMarketDataViaYahoo('AAPL','1-Dec-2001','31-Dec-2021','1mo'); 
P4 = getMarketDataViaYahoo('BA','1-Dec-2001','31-Dec-2021','1mo');  
P5 = getMarketDataViaYahoo('CAT','1-Dec-2001','31-Dec-2021','1mo'); 
P6 = getMarketDataViaYahoo('CSCO','1-Dec-2001','31-Dec-2021','1mo');
P7 = getMarketDataViaYahoo('CVX','1-Dec-2001','31-Dec-2021','1mo');
P8 = getMarketDataViaYahoo('GS','1-Dec-2001','31-Dec-2021','1mo');
P9 = getMarketDataViaYahoo('HD','1-Dec-2001','31-Dec-2021','1mo');
P10 = getMarketDataViaYahoo('HON','1-Dec-2001','31-Dec-2021','1mo');

PP = getMarketDataViaYahoo('PFE','1-Dec-2001','31-Dec-2021','1mo');

R1 = 100*(P1.Close(2:end) - P1.Close(1:end-1)) ./ P1.Close(1:end-1);
R2 = 100*(P2.Close(2:end) - P2.Close(1:end-1)) ./ P2.Close(1:end-1);
R3 = 100*(P3.Close(2:end) - P3.Close(1:end-1)) ./ P3.Close(1:end-1);
R4 = 100*(P4.Close(2:end) - P4.Close(1:end-1)) ./ P4.Close(1:end-1);
R5 = 100*(P5.Close(2:end) - P5.Close(1:end-1)) ./ P5.Close(1:end-1);
R6 = 100*(P6.Close(2:end) - P6.Close(1:end-1)) ./ P6.Close(1:end-1);
R7 = 100*(P7.Close(2:end) - P7.Close(1:end-1)) ./ P7.Close(1:end-1);
R8 = 100*(P8.Close(2:end) - P8.Close(1:end-1)) ./ P8.Close(1:end-1);
R9 = 100*(P9.Close(2:end) - P9.Close(1:end-1)) ./ P9.Close(1:end-1);
R10 = 100*(P10.Close(2:end) - P10.Close(1:end-1)) ./ P10.Close(1:end-1);

RP = 100*(PP.Close(2:end) - PP.Close(1:end-1)) ./ PP.Close(1:end-1);

% Standardize
R1 = (R1 - mean(R1))/std(R1);
R2 = (R2 - mean(R2))/std(R2);
R3 = (R3 - mean(R3))/std(R3);
R4 = (R4 - mean(R4))/std(R4);
R5 = (R5 - mean(R5))/std(R5);
R6 = (R6 - mean(R6))/std(R6);
R7 = (R7 - mean(R7))/std(R7);
R8 = (R8 - mean(R8))/std(R8);
R9 = (R9 - mean(R9))/std(R9);
R10 = (R10 - mean(R10))/std(R10);

RP = (RP - mean(RP))/std(RP);
dates = P1.Date;
dates2 = dates(2:end,:);

T = readtable('F-F_Research_Data_Factors.CSV','Range','B911:E1150','ReadVariableNames',false);
RM = T.Var1;
SMB = T.Var2;
HML = T.Var3;


% PCA
R = [R1,R2,R3,R4,R5,R6,R7,R8,R9,R10];
[B,F,var] = pca(R);
fprintf('\n');
fprintf('PCA eigenvalues\n');
disp(var);
fprintf('\n');
fprintf('PCA eigenvectors (first 2)\n');
disp(B(:,1:2));
F1 = F(:,1);
F2 = F(:,2);
fprintf('\n');
fprintf('Corr(F1,RM)    %.3f\n' , corr(F1,RM));
fprintf('Corr(F2,RM)    %.3f\n' , corr(F2,RM));
fprintf('Corr(F1,SMB)    %.3f\n' , corr(F1,SMB));
fprintf('Corr(F2,SMB)    %.3f\n' , corr(F2,SMB));
fprintf('Corr(F1,HML)    %.3f\n' , corr(F1,HML));
fprintf('Corr(F2,HML)    %.3f\n' , corr(F2,HML));
fprintf('\n');
fprintf('Pfizer Regression on PCA factors\n');
hac([F1,F2],RP,'type','HC','weights','HC3','display','full');
fprintf('\n');

% Factor
[Bf,sig,T,stats,FF] = factoran(R,2);
fprintf('\n');
fprintf('Factor Analysis Beta\n');
disp(Bf);
fprintf('\n');
fprintf('variances\n');
disp(sig);
fprintf('\n');
fprintf('Pfizer Regression on factors\n');
hac(FF,RP,'type','HC','weights','HC3','display','full');
fprintf('\n');
