%% run_fault_sweep.m
% trigger-based windows instead of the true fault time,
% CT/VT gain errors + sensor noise, (3) phasor impedance (R, X) features,
% independent out-of-distribution test set (DATASET = 'ood'),
% fault resistance = total fault resistance (ground resistance fixed at 0.01 ohm).

clear; clc;
DATASET = 'ood';   % 'train' = training set,  'ood' = independent out-of-distribution test set
if strcmp(DATASET,'ood'), rng(7); else, rng(42); end

%% ---------- USER SETTINGS ----------
mdl        = 'Transmission_Line_Model1';   
N_PER_CLASS = 5;      
N_NORMAL    = 20;     % no-fault runs (class 0)
USE_LOAD_VARIATION = true;
loadBlock   = [mdl '/LoadBlock'];    

f0   = 60;            % Hz
Ltot = 40;            % km
Rkm  = 0.15;          % ohm/km
Lkm  = 1.3263e-3;     % H/km
P0   = 304.8e6;       % W   (381 MVA, 0.8 pf)
Q0   = 228.6e6;       % VAr
Vnom = 220e3;         % V line-to-line

Nspc = 360;                       % samples per cycle
Ts   = 1/(f0*Nspc);               % fixed step (integer samples/cycle)
Tstop = 0.25;                     % s

FULL_RUN = true;     % false = quick test, true = real dataset
if FULL_RUN
    if strcmp(DATASET,'ood'), N_PER_CLASS = 100; N_NORMAL = 100;   
    else,                     N_PER_CLASS = 300; N_NORMAL = 300; end 
    outFile = ['fault_dataset_' DATASET '.csv'];
else
    outFile = ['fault_dataset_' DATASET '_quicktest.csv'];
end

RfMin = 0.01;                          % fault resistance (ohm)
if strcmp(DATASET,'ood')
    % Conditions the training set never covered
    dMin = 0.025*Ltot; dMax = 0.975*Ltot;   % 1 - 39 km (training: 2 - 38)
    RfMax = 100;                            % training: 50
    P_RANGE = [0.5 1.3]; Q_RANGE = [0.5 1.5]; % P and Q vary independently -power factor changes
    NOISE = 1.5;                            % 1.5x the training measurement errors
else
    dMin = 0.05*Ltot; dMax = 0.95*Ltot;
    RfMax = 50;
    P_RANGE = [0.8 1.2]; Q_RANGE = 'same';  % P and Q scale together (power factor stays 0.8)
    NOISE = 1;
end

% Fault classes: name, list of [A B C G] combinations
classes = {
 'LG',  [1 0 0 1; 0 1 0 1; 0 0 1 1];
 'LL',  [1 1 0 0; 0 1 1 0; 1 0 1 0];
 'LLG', [1 1 0 1; 0 1 1 1; 1 0 1 1];
 'LLL', [1 1 1 0];
};

%% ---------- MODEL SETUP ----------
load_system(mdl);
set_param(mdl,'SolverType','Fixed-step','Solver','FixedStepAuto', ...
    'FixedStep',sprintf('%.15g',Ts),'StopTime',num2str(Tstop));
set_param([mdl '/powergui'],'SimulationMode','Discrete', ...
    'SampleTime',sprintf('%.15g',Ts));
set_param([mdl '/Line1'],'BranchType','RL');
set_param([mdl '/Line2'],'BranchType','RL');
% If a parameter name errors, run: get_param([mdl '/Fault'],'DialogParameters')

%% ---------- BUILD RUN LIST ----------
runs = struct('cls',{},'clsName',{},'phases',{});
for c = 1:size(classes,1)
    combos = classes{c,2};
    for k = 1:N_PER_CLASS
        row = combos(randi(size(combos,1)),:);
        runs(end+1) = struct('cls',c,'clsName',classes{c,1},'phases',row); %#ok<SAGROW>
    end
end
for k = 1:N_NORMAL
    runs(end+1) = struct('cls',0,'clsName','NF','phases',[0 0 0 0]); %#ok<SAGROW>
end
runs = runs(randperm(numel(runs)));   % shuffle

%% ---------- MAIN LOOP ----------
rows = {}; nOK = 0; examples = struct();
for i = 1:numel(runs)
    r = runs(i);
    d  = dMin + (dMax-dMin)*rand;                 % km
    Rf = exp(log(RfMin) + (log(RfMax)-log(RfMin))*rand);  % log-uniform
    tf = 0.12 + 0.03*rand;  tf = round(tf/Ts)*Ts; % inception time (s)

    % Line split
    set_param([mdl '/Line1'],'Resistance',num2str(Rkm*d,'%.6g'), ...
        'Inductance',num2str(Lkm*d,'%.6g'));
    set_param([mdl '/Line2'],'Resistance',num2str(Rkm*(Ltot-d),'%.6g'), ...
        'Inductance',num2str(Lkm*(Ltot-d),'%.6g'));

    % Fault block
    onoff = {'off','on'};
    set_param([mdl '/Fault'], ...
        'FaultA',onoff{r.phases(1)+1}, 'FaultB',onoff{r.phases(2)+1}, ...
        'FaultC',onoff{r.phases(3)+1}, 'GroundFault',onoff{r.phases(4)+1}, ...
        'FaultResistance',num2str(Rf,'%.6g'), 'GroundResistance','0.01', ...
        'External','off');
    if r.cls == 0
        set_param([mdl '/Fault'],'SwitchTimes','[100]');   
        set_param([mdl '/Fault'],'FaultA','on');           
    else
        set_param([mdl '/Fault'],'SwitchTimes',sprintf('[%.9g]',tf));
    end

    % Load variation
    if USE_LOAD_VARIATION
        sP = P_RANGE(1) + diff(P_RANGE)*rand;
        if ischar(Q_RANGE), sQ = sP; else, sQ = Q_RANGE(1) + diff(Q_RANGE)*rand; end
        set_param(loadBlock,'ActivePower',num2str(P0*sP), ...
            'InductivePower',num2str(Q0*sQ));
    end

    try
        out = sim(mdl,'ReturnWorkspaceOutputs','on');
    catch ME
        if nOK == 0, rethrow(ME); end   
        warning('Run %d failed: %s', i, ME.message); continue;
    end
    nOK = nOK + 1;
    V = getsig(out, {'Vabc_s','simout'});
    I = getsig(out, {'Iabc_s','simout1'});

    [V, I] = add_measurement_errors(V, I, NOISE);
    [feat, trig_ms, trig_ok] = extract_features(V, I, tf, Ts, Nspc, r.cls);
    if ~isfield(examples, r.clsName)     
        examples.(r.clsName) = struct('V',V,'I',I,'tf',tf,'d',d,'Rf',Rf,'Ts',Ts);
    end
    rows{end+1} = [r.cls, r.phases, d, Rf, tf, trig_ok, trig_ms, feat]; %#ok<SAGROW>
    if r.cls == 0, rows{end}(6) = NaN; end   

    if mod(i,25)==0
        fprintf('%d / %d done\n', i, numel(runs));
        save('fault_checkpoint.mat','rows');
    end
end

%% ---------- SAVE ----------
if isempty(rows), error('No successful runs, nothing to save.'); end
M = vertcat(rows{:});
featNames = feature_names();
varNames = [{'class','pA','pB','pC','pG','d_km','Rf_ohm','tf_s','triggered','trig_delay_ms'}, featNames];
T = array2table(M,'VariableNames',varNames);
names = {'NF','LG','LL','LLG','LLL'};
T.class_name = names(T.class+1).';
writetable(T,outFile);
save('fault_examples.mat','examples');
fprintf('Saved %d rows to %s\n', height(T), outFile);

%% ---------- LOCAL FUNCTIONS ----------
function X = getsig(out, names)
    % Reads a To Workspace signal by variable name; accepts Array or Timeseries.
    for k = 1:numel(names)
        try, v = out.get(names{k}); catch, continue; end
        if isempty(v), continue; end
        if isa(v,'timeseries'), v = v.Data; end
        X = squeeze(v);
        if size(X,1) == 3 && size(X,2) > 3, X = X.'; end
        return;
    end
    error('Signal not found. Check the To Workspace variable names.');
end

function names = feature_names()
    names = {'Va','Vb','Vc','Ia','Ib','Ic', ...
             'dVa','dVb','dVc','dIa','dIb','dIc', ...
             'V0','V1','V2','I0','I1','I2', ...
             'I0_I1','I2_I1','V1_I1','V2_I2','V0_I0', ...
             'Rpa','Rpb','Rpc','Xpa','Xpb','Xpc', ...
             'Rab','Rbc','Rca','Xab','Xbc','Xca','R1','X1'};
end

function [V, I] = add_measurement_errors(V, I, k)
    % Instrument transformer ratio errors (fixed per run) + sensor noise.
    % k scales all of it (1 = training, 1.5 = out-of-distribution set).
    gV = 1 + k*0.005*(2*rand(1,3)-1);            % VT error, +-0.5 %
    gI = 1 + k*0.010*(2*rand(1,3)-1);            % CT error, +-1 %
    V = V.*gV + k*0.002*204e3*randn(size(V));    % noise: 0.2 % of nominal peak voltage
    I = I.*gI + k*0.002*1414*randn(size(I));     % noise: 0.2 % of nominal peak current
end

function it = find_trigger(I, N, thr)
    % First sample where the 1-cycle RMS of any phase current is thr times its value
    % one cycle earlier. uses no knowledge of the true fault time.
    r  = sqrt(movmean(I.^2, [N-1 0], 1));        % causal 1-cycle RMS
    k0 = max(2*N, round(0.10*N*60));             % ignore start-up transients (t < 0.1 s)
    k  = k0:size(I,1);
    ratio = max(r(k,:) ./ r(k-N,:), [], 2);
    j  = find(ratio > thr, 1, 'first');
    if isempty(j), it = NaN; else, it = k(j); end
end

function [f, trig_ms, ok] = extract_features(V, I, tf, Ts, N, cls)
    nF = numel(feature_names());
    it = find_trigger(I, N, 1.3);
    ok = ~isnan(it); trig_ms = NaN;
    if ok
        i0 = it;
        if cls > 0, trig_ms = (it - (round(tf/Ts)+1))*Ts*1e3; end   % detection delay (ms)
    elseif cls == 0
        i0 = round(tf/Ts) + 1;                   % no-fault run: arbitrary pseudo-event time
    else
        f = nan(1,nF); return;                   % fault that was never detected
    end
    pre  = (i0-2*N+1):(i0-N);                    
    post = (i0+1):(i0+N);                        
    Pv0 = phasors(V(pre,:));  Pi0 = phasors(I(pre,:));
    Pv1 = phasors(V(post,:)); Pi1 = phasors(I(post,:));
    a = exp(1j*2*pi/3);
    seq = @(P) [ (P(1)+P(2)+P(3))/3, ...
                 (P(1)+a*P(2)+a^2*P(3))/3, ...
                 (P(1)+a^2*P(2)+a*P(3))/3 ];
    svc = seq(Pv1); sic = seq(Pi1);
    sv = abs(svc); si = abs(sic);                % [zero pos neg]
    e = 1e-6;
    Zph = Pv1 ./ Pi1;                            % per-phase apparent impedance
    Zll = [(Pv1(1)-Pv1(2))/(Pi1(1)-Pi1(2)), ...  % phase-to-phase loop impedances
           (Pv1(2)-Pv1(3))/(Pi1(2)-Pi1(3)), ...
           (Pv1(3)-Pv1(1))/(Pi1(3)-Pi1(1))];
    Z1  = svc(2)/sic(2);                         % positive-sequence impedance
    f = [abs(Pv1), abs(Pi1), ...
         abs(Pv1)./abs(Pv0), abs(Pi1)./abs(Pi0), ...
         sv, si, ...
         si(1)/(si(2)+e), si(3)/(si(2)+e), ...
         sv(2)/(si(2)+e), sv(3)/(si(3)+e), sv(1)/(si(1)+e), ...
         real(Zph), imag(Zph), real(Zll), imag(Zll), real(Z1), imag(Z1)];
end

function P = phasors(x)           % x: N x 3, exactly one cycle
    N = size(x,1); n = (0:N-1).';
    w = exp(-1j*2*pi*n/N);
    P = sqrt(2)/N * (w.' * x);    
end