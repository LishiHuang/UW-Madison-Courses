% Topic 2: Linear Models

clear all

% GE
GE = getMarketDataViaYahoo('GE','1-Jan-1962','31-Dec-2023','1d');
price = GE.Close;
logprice = log(price);
dates = GE.Date;
r = 100*(logprice(2:end) - logprice(1:end-1));

figure(1);
plot(dates,price);
ylabel('Dollars')
title('Price of GE Stock');
print('Topic2_1','-dpdf');
box off

dates = dates(2:end,:);
figure(2);
plot(dates,r);
ylabel('Percentage Return')
title('Return on GE Stock');
print('Topic2_2','-dpdf');
box off

% Q4
d = day(dates,'dayofweek');
Xd = dummyvar(categorical(d));
Xd = Xd(:,1:4);
fprintf('Regression Q4s\n');
hac(Xd,r,'type','HC','weights','HC3','display','full');

% Q5
d1 = day(dates,'dayofmonth');
d2 = day(dates,'dayofweek');
condition1=(d1==1);
condition2=(d1==31);
condition3=(d1==31)&(d2==6);
d = zeros(size(d1));
d(condition1) =1;
d(condition2) =2;
d(condition3) =3;
Dd = dummyvar(categorical(d));
Dd = Dd(:,1:3);
fprintf('Regression Q5\n');
hac(Dd,r,'type','HC','weights','HC3','display','full');

%Q6
time = (1:length(r))';
time_trend=time/max(time);
fprintf('Regression Q6\n');
hac(time_trend,r,'type','HC','weights','HC3','display','full');

% Q7
L = 4;
Y = r(L+1:end);
X = [r(L:end-1),r(L-1:end-2),r(L-2:end-3),r(L-3:end-4)];
fprintf('Regression with AR(4)\n');
hac(X,Y,'type','HC','weights','HC3','display','full');

% Q8
condition4=double((r>0));
Y8 = r(L+1:end);
X8= condition4(L:end-1);
fprintf('Regression Q8\n');
hac(X8,Y8,'type','HC','weights','HC3','display','full');

% Q9
fprintf('\n')
DeltaY = diff(logprice);
[h,p,t]=adftest(DeltaY,'model','TS','lags',6);
fprintf('6 lags    %.3f\n',[h,p,t])
%{
% Regression with lagged volume (in 10,000,000 units) as regressor
volume = GE.Volume;
volume = volume(1:end-1)/10000000;
fprintf('Regression with volume\n');
hac(volume,r,'type','HC','weights','HC3','display','full');

% Regression with log of lagged volume (in 10,000,000 units) as regressor
logV = log(volume);
hac(logV,r,'type','HC','weights','HC3','display','full');

%}