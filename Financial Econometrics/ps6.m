% ps 6

clear all

% Stock prices and returns
P1 = getMarketDataViaYahoo('NKE','1-Jan-2000','31-Dec-2021','1d'); % Nike
P2 = getMarketDataViaYahoo('SBUX','1-Jan-2000','31-Dec-2021','1d'); % Starbucks
dates = P1.Date;

R1 = 100*(P1.Close(2:end) - P1.Close(1:end-1)) ./ P1.Close(1:end-1);
R2 = 100*(P2.Close(2:end) - P2.Close(1:end-1)) ./ P2.Close(1:end-1);
dates2 = dates(2:end,:);

% 2-asset portfolio
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
fprintf('Nike & Starbucks portfolio\n');
fprintf('mean Nike return  	   %.3f\n' , mu1);
fprintf('mean Starbucks return  	   %.3f\n' , mu2);
fprintf('SD Nike return  	   %.3f\n' , sd1);
fprintf('SD Starbucks return  	   %.3f\n' , sd2);
fprintf('Correlation	  	   %.3f\n' , rho);
fprintf('Cov Matrix	  	  \n') ;
disp(V);
fprintf('Weight on Starbucks  	   %.3f\n' , w1);
fprintf('Weight on Nike  	   %.3f\n' , 1-w1);
fprintf('SD of portfolio  	   %.3f\n' , s1);
fprintf('\n');

w = (0:.01:1);
sd = sqrt((w.^2)*sig2 + ((1-w).^2)*sig1 + 2*w.*(1-w)*sig12);

figure(1);
plot(w,sd);
xlabel('Portfolio weight w');
ylabel('Standard Deviation');
title('Standard Deviation as Function of Portfolio Weight');
hold on
plot(w1,s1,'r.','MarkerSize', 20);
hold off
print('Topic6_1','-dpdf');
box off


