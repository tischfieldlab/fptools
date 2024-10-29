%% Data importation %%
clear all;
clc;
[path] = uigetdir;
cd(path);
SDKpath = 'C:\Users\Arlene\Documents\Margolis Lab\Fiber photometry\TDTMatlabSDK-master';
addpath(genpath(SDKpath));

%Automatically loads in the AUC/AUC_BOOTSTRAP functions for the AUROC
%analysis
aurocpath = 'C:\Users\Arlene\Documents\Margolis Lab\Fiber photometry\AUC';
addpath(genpath(aurocpath));




%% Fiber photometry analysis %%


%Import data file and segment out necessary information
Data = TDTbin2mat(path);
EventsFlag = str2num(cell2mat(inputdlg('Does this file contain event marks? If yes, input 1: ')));
GCaMP = Data.streams.G__B.data(8000:end).'; %Raw GCaMP signals
Iso = Data.streams.IsoB.data(8000:end).'; %Raw Isosbestic signals
Fs = Data.streams.IsoB.fs; %Frame acqusition sampling
Ts = ((1:numel(Data.streams.G__B.data(1,8000:end))) / Fs)'; %Calculate timestamps by Fs
if EventsFlag == 1
    Events2 = Data.epocs.U13_.onset;  %TTL (button/click) events

else
end


    %Construct list for detrending via line of best fit
    list={GCaMP,Iso};
    for i=1:length(list)
       BestFitG= polyfit(Ts,list{i},2);
       TLG = ((BestFitG(1,1)*(Ts.^2)) + (BestFitG(1,2)*Ts) + (BestFitG(1,3)));
       DetrendSignals{i} = list{i}-TLG;
       clear TLG BestFitG
    end
    
    %Normalize the values for Z score and MAD Z score calculations
    meanBaseG = mean(DetrendSignals{1});
    medianBaseG = median(DetrendSignals{1});
    madBaseG = mean(mad(DetrendSignals{1}));
    stdBaseG = std(DetrendSignals{1});
    meanBaseI = mean(DetrendSignals{2});
    medianBaseI = median(DetrendSignals{2});
    madBaseI = mean(mad(DetrendSignals{2}));
    stdBaseI = std(DetrendSignals{2});
    
    ZGCaMP = (DetrendSignals{1} - meanBaseG)/stdBaseG;
    ZIso = (DetrendSignals{2} - meanBaseI)/stdBaseI;
    ZmadGCaMP = (DetrendSignals{1} - medianBaseG)/madBaseG;
    ZmadIso = (DetrendSignals{2} - medianBaseI)/madBaseI;

    %Subtract the isosbestic signal from the GCaMP signal
    Z = ZmadGCaMP - ZmadIso;
    
    %Define an x vector that goes from -second to + the second
    %Convert the timestamps into sampling frequency
    Seconds= 30;
    Before=round(Seconds*Fs);
    After=round(Seconds*Fs);

%Call U13 stimulus
    if EventsFlag == 1
        for i=2:length(Events2)-1   
            a=find(Events2(i)< Ts(:,1));
            a=a(1,1);
            ZAlign2(i,:)= Z(a-Before:a+After,1);
        end

    ZAlignMean2 = mean(ZAlign2(2:end,:));
    ZAlignstd = std(ZAlign2(2:end,:));
    xlimits = length(ZAlign2);

    %plot window of time
    figure(1)
    subplot(2,1,2);
    plot(ZAlignMean2);
    xlim([0 xlimits]);
    ylim([-5 5]);
    xline(Before,'r','LineWidth',1)

    subplot(2,1,1)
    imagesc(ZAlign2);
    caxis([-2 3]);
    xlim([0 xlimits]);
    xline(Before,'r','LineWidth',1)
    title('Pinprick')
    EventOutcome = {};
    else
        plot(Z)
    end
