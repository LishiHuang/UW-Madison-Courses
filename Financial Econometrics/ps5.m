% PS 5

%Q1-a
NYT = getMarketDataViaYahoo('NYT','1-Jan-1985','31-Dec-2023','1d');
r = 100*(log(NYT.Close(2:end)) - log(NYT.Close(1:end-1)));

NYTG11=garch('GARCHLags',1,'ARCHLags',1,'Offset',NaN);
estNYT11=estimate(NYTG11,r);
summarize(estNYT11);
%Q1-b
NYTG23=garch('GARCHLags',1:2,'ARCHLags',1:3,'Offset',NaN);
estNYT23=estimate(NYTG23,r);
summarize(estNYT23);
%Q1-C
%{
AIC_matrix = zeros(10, 10);
BIC_matrix = zeros(10, 10);
for p = 1:10
    for q = 1:10
        % Fit the GARCH(p,q) model
        GARCH_model = garch('GARCHLags',1:p,'ARCHLags',1:q);
        est_GARCH = estimate(GARCH_model, r);
        results = summarize(est_GARCH); %store estimation results in a construct
        % Calculate AIC and BIC
        AIC_matrix(p, q) = results.AIC;
        BIC_matrix(p, q) = results.BIC;
    end
end

% Find the indices of the minimum AIC and BIC values
[min_AIC, ind_AIC] = min(AIC_matrix(:));
[min_BIC, ind_BIC] = min(BIC_matrix(:));

% Convert the linear indices to subscripts
[p_AIC, q_AIC] = ind2sub(size(AIC_matrix), ind_AIC);
[p_BIC, q_BIC] = ind2sub(size(BIC_matrix), ind_BIC);

disp('Best GARCH model according to AIC:');
disp(['p = ', num2str(p_AIC), ', q = ', num2str(q_AIC)]);
disp(['AIC = ', num2str(min_AIC)]);

disp('Best GARCH model according to BIC:');
disp(['p = ', num2str(p_BIC), ', q = ', num2str(q_BIC)]);
disp(['BIC = ', num2str(min_BIC)]);
%}
%Q1-d
mu=mean(r);
residuals=r-mu;
NYTG62=garch('GARCHLags',1:6,'ARCHLags',1:2,'Offset',NaN);
estNYT62=estimate(NYTG62,r);
summarize(estNYT62);
v62 = sqrt(infer(estNYT62,r));

% De-volatilized residuals
devol_residuals = residuals ./ v62;

% Plot the series
figure;
subplot(3,1,1);
plot(residuals);
title('Residuals');
subplot(3,1,2);
plot(v62);
title('Estimated Volatilities');
subplot(3,1,3);
plot(devol_residuals);
title('De-volatilized Residuals');

%Q1-e
figure;
histogram(devol_residuals, 'Normalization', 'pdf');
title('Density of De-volatilized Residuals');

%Q2-a
NYTG62T=garch('GARCHLags',1:6,'ARCHLags',1:2,'Offset',NaN,'Distribution','t');
estNYT62T=estimate(NYTG62T,r);
summarize(estNYT62T);

%Q2-b
% Extract the estimated degrees of freedom parameter
nu = estNYT62T.Distribution.DoF;

% Display the estimated degrees of freedom parameter
disp(['Estimated Degrees of Freedom (nu): ', num2str(nu)]);

%Q3-a
% Augment the preferred GARCH(p,q) model with a GJR leverage effect
GJR_model = gjr('GARCHLags', 1:6, 'ARCHLags', 1:2, 'LeverageLags',1:2,'Offset', NaN);
estimates_GJR = estimate(GJR_model, r);

% Display parameter estimates
disp('Parameter Estimates for GJR GARCH Model:');
disp(estimates_GJR);
