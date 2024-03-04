clear all;
close all;
clc;

addpath([pwd '/KPCA/']);
addpath([pwd '/../dataset/BCI4_2B/']);
addpath([pwd '/../dataset/BCI3_3B/']);

fs=250;
[s_sess1,h_sess1]=sload('B0201T.gdf'); % B0401T dataset preparation
Trials_All_Ind_sess1=find(h_sess1.EVENT.TYP==768);
Reject_Ind_sess1=find(h_sess1.ArtifactSelection==1);

Trials_All_Ind_sess1(Reject_Ind_sess1)=[];
h_sess1.Classlabel(Reject_Ind_sess1)=[];

Trials_Ind_sess1=Trials_All_Ind_sess1;
Z_sess1= h_sess1.Classlabel';
trial=1;
for i=1:length(Trials_Ind_sess1)
    st_sess1=h_sess1.EVENT.POS(Trials_Ind_sess1(i));
    dur_sess1=h_sess1.EVENT.DUR(Trials_Ind_sess1(i));
    Y_temp_sess1(:,:)=s_sess1(st_sess1:st_sess1+dur_sess1-1,1:3);
    if(find(isnan(Y_temp_sess1(:,:)))~=0) %Y_temp=[3x350x280], 3channels, 350samples, 280trials
       i
       Z_sess1(trial)=[];
    else
        Y_sess1(:,:,trial)=Y_temp_sess1(:,:)';
        trial=trial+1;
    end
end

[s_sess2,h_sess2]=sload('B0202T.gdf'); % B0402T dataset preparation
Trials_All_Ind_sess2=find(h_sess2.EVENT.TYP==768);
Reject_Ind_sess2=find(h_sess2.ArtifactSelection==1);

Trials_All_Ind_sess2(Reject_Ind_sess2)=[];
h_sess2.Classlabel(Reject_Ind_sess2)=[];

Trials_Ind_sess2=Trials_All_Ind_sess2;
Z_sess2= h_sess2.Classlabel';
trial=1;
for i=1:length(Trials_Ind_sess2)
    st_sess2=h_sess2.EVENT.POS(Trials_Ind_sess2(i));
    dur_sess2=h_sess2.EVENT.DUR(Trials_Ind_sess2(i));
    Y_temp_sess2(:,:)=s_sess2(st_sess2:st_sess2+dur_sess2-1,1:3);
    if(find(isnan(Y_temp_sess2(:,:)))~=0) 
       i
       Z_sess2(trial)=[];
    else
        Y_sess2(:,:,trial)=Y_temp_sess2(:,:)';
        trial=trial+1;
    end
end

[s_sess3,h_sess3]=sload('B0203T.gdf'); % B0403T dataset preparation
Trials_All_Ind_sess3=find(h_sess3.EVENT.TYP==768);
Reject_Ind_sess3=find(h_sess3.ArtifactSelection==1);

Trials_All_Ind_sess3(Reject_Ind_sess3)=[];
h_sess3.Classlabel(Reject_Ind_sess3)=[];

Trials_Ind_sess3=Trials_All_Ind_sess3;
Z_sess3= h_sess3.Classlabel';
trial=1;
for i=1:length(Trials_Ind_sess3)
    st_sess3=h_sess3.EVENT.POS(Trials_Ind_sess3(i));
    dur_sess3=h_sess3.EVENT.DUR(Trials_Ind_sess3(i));
    Y_temp_sess3(:,:)=s_sess3(st_sess3:st_sess3+1999,1:3);
    if(find(isnan(Y_temp_sess3(:,:)))~=0) 
       i
       Z_sess3(trial)=[];
    else
        Y_sess3(:,:,trial)=Y_temp_sess3(:,:)';
        trial=trial+1;
    end
end

[s_sess4,h_sess4]=sload('B0204E.gdf'); % B0403T dataset preparation
B0404E_Labels=load('B0204E.mat');

Trials_All_Ind_sess4=find(h_sess4.EVENT.TYP==768);
Reject_Ind_sess4=find(h_sess4.ArtifactSelection==1);

Trials_All_Ind_sess4(Reject_Ind_sess4)=[];
B0404E_Labels.classlabel(Reject_Ind_sess4)=[];

Trials_Ind_sess4=Trials_All_Ind_sess4;
Z_sess4= B0404E_Labels.classlabel';
trial=1;
for i=1:length(Trials_Ind_sess4)
    clear Y_temp;
    st_sess4=h_sess4.EVENT.POS(Trials_Ind_sess4(i));
    dur_sess4=h_sess4.EVENT.DUR(Trials_Ind_sess4(i));
    Y_temp_sess4(:,:)=s_sess4(st_sess4:st_sess4+1999,1:3);
    if(find(isnan(Y_temp_sess4(:,:)))~=0) 
       i
       Z_sess4(trial)=[];
    else
        Y_sess4(:,:,trial)=Y_temp_sess4(:,:)';
        trial=trial+1;
    end
end

[s_sess5,h_sess5]=sload('B0205E.gdf'); % B0403T dataset preparation
B0405E_Labels=load('B0205E.mat');

Trials_All_Ind_sess5=find(h_sess5.EVENT.TYP==768);
Reject_Ind_sess5=find(h_sess5.ArtifactSelection==1);

Trials_All_Ind_sess5(Reject_Ind_sess5)=[];
B0405E_Labels.classlabel(Reject_Ind_sess5)=[];

Trials_Ind_sess5=Trials_All_Ind_sess5;
Z_sess5= B0405E_Labels.classlabel';
trial=1;
for i=1:length(Trials_Ind_sess5) 
    clear Y_temp;
    st_sess5=h_sess5.EVENT.POS(Trials_Ind_sess5(i));
    dur_sess5=h_sess5.EVENT.DUR(Trials_Ind_sess5(i));
    Y_temp_sess5(:,:)=s_sess5(st_sess5:st_sess5+1999,1:3);
    if(find(isnan(Y_temp_sess5(:,:)))~=0) 
       i
       Z_sess5(trial)=[];
    else
        Y_sess5(:,:,trial)=Y_temp_sess5(:,:)';
        trial=trial+1;
    end
end

Ch=3;
ZTn=[Z_sess1 Z_sess2 Z_sess3];
ZTt=Z_sess5;
YTn=cat(3,Y_sess1,Y_sess2,Y_sess3);
YTt=Y_sess5;
dmax=2*Ch;

indL=find(ZTn==1);
indR=find(ZTn==2);
NL=length(indL);
NR=length(indR);
L=min([NL,NR]);
indTt_L=find(ZTt==1);
indTt_R=find(ZTt==2);
NTest=length(ZTt);

%% i. PCA-SM
st=3.5*fs+1:1:5*fs+1;
ed=5.5*fs:1:7*fs;
NRow=1;
for d=1:dmax-1
    clear W;
    for epoch=1:length(st)
        st_time=floor(st(epoch));
        ed_time=floor(ed(epoch));

        % Feature calculation
        clear BP_Temp; clear BPL;
        for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPL_Temp=[];
            for c=1:Ch
                BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
            end
            BPL(:,trial)=BPL_Temp;
        end

        clear BP_Temp; clear BPR;
        for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];    % For BCI4_2B dataset 
            BPR_Temp=[];
            for c=1:Ch
                BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
            end
            BPR(:,trial)=BPR_Temp;
        end

        t=d;
        Rr=cov(BPR');  Rl=cov(BPL'); 
        [Ur Er ~]=svd(Rr); [Ul El ~]=svd(Rl); 

        % Measuring orthogonality among classes of training data
        clear Url; clear Erl; clear Srl;
        [Url Erl ~]=svd(Ur(:,1:d)'*Ul(:,1:d)); cos_theta=diag(Erl);
        Srl=sum(cos_theta(1:t).^2)/t;

        W(epoch)=1-(Srl);
    end

    [VV II]=max(W);
    st_time=floor(st(II));
    ed_time=floor(ed(II));

    % optimum Linear Subspace model
    for trial=1:L
       Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
       BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])]; 
       BPL_Temp=[];
       for c=1:Ch
           BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
       end
       BPL(:,trial)=BPL_Temp;
     end

     for trial=1:L
       Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
       BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
       BPR_Temp=[];
       for c=1:Ch
          BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
       end
       BPR(:,trial)=BPR_Temp;
     end

     Rr=cov(BPR');  Rl=cov(BPL'); 
     [Ur Er ~]=svd(Rr); [Ul El ~]=svd(Rl); 

    for epoch=1:length(st)
       st_time=floor(st(epoch));
       ed_time=floor(ed(epoch));
       for trial=1:NTest
           Ytemp(:,:)=YTt(:,st_time:ed_time,trial);
           BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
           BPTest_Temp=[];
           for c=1:Ch
               BPTest_Temp=[BPTest_Temp;BP_Temp(:,c)];
           end
           BPTest(:,trial)=BPTest_Temp;
       end
       % SM testing
       clear Zpredict;
       for trial=1:NTest
           RTt=BPTest(:,trial)*BPTest(:,trial)';
           [UTt ETt ~]=svd(RTt);
           clear cos_theta; clear ETt_l; clear STt_l;
           [~,ETt_l,~]=svd(UTt(:,1:d)'*Ul(:,1:d));    cos_theta=diag(ETt_l);
           STt_l=sum(cos_theta(1:t).^2)/t;
           clear cos_theta; clear ETt_r; clear STt_r;
           [~,ETt_r,~]=svd(UTt(:,1:d)'*Ur(:,1:d));    cos_theta=diag(ETt_r);
           STt_r=sum(cos_theta(1:t).^2)/t;
           [VV,Zpredict(trial)]=max([STt_l,STt_r]);
       end
       [Cf order]=confusionmat(ZTt,Zpredict');
       Acc(epoch)=(Cf(1,1)+Cf(2,2))/NTest;  
       Cf_mat(:,:,epoch)=Cf;
    end
    [Vend Iend]=max(Acc);
    stats=statsOfMeasure(Cf_mat(:,:,Iend),1); % Statistics was measured only for optimum parameters 
    Result_PCA_SM(NRow,:)=[d W(II) II max(Acc)]
    NRow=NRow+1;
end

%% ii. PCA-DSM
clear st; clear ed;
st=3.5*fs+1:10:5*fs+1;
ed=5.5*fs:10:7*fs;
NROW=1;
for d=1:dmax-1
    clear WDS;
    for epoch=1:length(st)
        st_time=floor(st(epoch));
        ed_time=floor(ed(epoch));

        % Feature calculation
        clear BP_Temp; clear BPL;
        for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPL_Temp=[];
            for c=1:Ch
                BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
            end
            BPL(:,trial)=BPL_Temp;
        end

        clear BP_Temp; clear BPR;
        for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPR_Temp=[];
            for c=1:Ch
                BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
            end
            BPR(:,trial)=BPR_Temp;
        end

       clear Rr; clear Rl; clear Ur; clear Ul; clear Er; clear El;
       Rr=cov(BPR');  Rl=cov(BPL'); 
       [Ur Er ~]=svd(Rr); [Ul El ~]=svd(Rl); 

       P=0;Q=0;
       for i=1:d
           P=P+Ur(:,i)*Ur(:,i)';
           Q=Q+Ul(:,i)*Ul(:,i)';
       end

       clear UDS; clear EDS;
       [UDS,EDS,~]=svd(P+Q);

       D=UDS(:,find(diag(EDS)<1));
       dDs_max=length(find(diag(EDS)<1));
       clear BPLDS; clear BPRDS; clear RrDS; clear RlDS; clear UrDS; clear ErDS; clear UlDS; clear ErDS;
       BPLDS=D'*BPL; BPRDS=D'*BPR;  
       RrDS=cov(BPRDS'); RlDS=cov(BPLDS'); 
       [UrDS ErDS ~]=svd(RrDS);   [UlDS ElDS ~]=svd(RlDS);

       % Measuring orthogonality among classes of training data
       for dDs=1:dDs_max
            clear cos_theta; clear ErlDS; tDs=dDs;
            [~,ErlDS,~]=svd(UrDS(:,1:dDs)'*UlDS(:,1:dDs));  cos_theta=diag(ErlDS);
            SrlDS=sum(cos_theta(1:tDs).^2)/tDs;
            WDS(epoch,dDs)=1-SrlDS;
       end
    end

    [M,I] = max(WDS,[],"all","linear");
    [epoch_opt, dDs] = ind2sub(size(A),I)
    st_time=floor(st(epoch_opt));
    ed_time=floor(ed(epoch_opt));
    % Optimum model
    clear BP_Temp; clear BPL;
    for trial=1:L
        Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
        BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
        BPL_Temp=[];
        for c=1:Ch
            BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
        end
        BPL(:,trial)=BPL_Temp;
    end            

    clear BP_Temp; clear BPR;
    for trial=1:L
        Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
        BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
        BPR_Temp=[];
        for c=1:Ch
            BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
        end
        BPR(:,trial)=BPR_Temp;
    end

    clear Rr; clear Rl; clear Ur; clear Er; clear Ul; clear El;
    Rr=cov(BPR');  Rl=cov(BPL'); 
    [Ur Er ~]=svd(Rr); [Ul El ~]=svd(Rl); 

    P=0;Q=0;
    for i=1:d
        P=P+Ur(:,i)*Ur(:,i)';
        Q=Q+Ul(:,i)*Ul(:,i)';
    end

    clear UDS; clear EDS;
    [UDS,EDS,~]=svd(P+Q);

    clear D; clear BPLDS; clear BPRDS;clear RrDS; clear RlDS; clear UrDS; clear UlDS; clear ElDS; clear ErDS;
    tDs=dDs; 
    D=UDS(:,find(diag(EDS)<1));

    BPLDS=D'*BPL; BPRDS=D'*BPR;  
    RrDS=cov(BPRDS'); RlDS=cov(BPLDS'); 
    [UrDS ErDS ~]=svd(RrDS);   [UlDS ElDS ~]=svd(RlDS);

    for epoch=1:length(st)
        st_time=floor(st(epoch));
        ed_time=floor(ed(epoch));
        clear BP_Temp; clear BPTest;
        for trial=1:NTest
            Ytemp(:,:)=YTt(:,st_time:ed_time,trial);
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPTest_Temp=[];
            for c=1:Ch
                BPTest_Temp=[BPTest_Temp;BP_Temp(:,c)];
            end
            BPTest(:,trial)=BPTest_Temp;
        end
        clear BPTestDS;
        BPTestDS=D'*BPTest;
        % SM testing
        clear Zpredict;
        for trial=1:NTest
            clear RTtDS; clear UTtDS; clear ETtDS;
            RTtDS=BPTestDS(:,trial)*BPTestDS(:,trial)';
            [UTtDS ETtDS ~]=svd(RTtDS);

            clear cos_theta; clear ETt_lDS; clear ETt_rDS;
            [~,ETt_lDS,~]=svd(UTtDS(:,1:dDs)'*UlDS(:,1:dDs));    cos_theta=diag(ETt_lDS);
            STt_lDS=sum(cos_theta(1:tDs).^2)/tDs;

            clear cos_theta;
            [~,ETt_rDS,~]=svd(UTtDS(:,1:dDs)'*UrDS(:,1:dDs));    cos_theta=diag(ETt_rDS);
            STt_rDS=sum(cos_theta(1:tDs).^2)/tDs;
            [VV,Zpredict(trial)]=max([STt_lDS,STt_rDS]);
        end
        [Cf order]=confusionmat(ZTt,Zpredict);
        Acc=(Cf(1,1)+Cf(2,2))/NTest;
        Result_PCA_DSM(NROW,:)=[d dDs epoch Acc]
        Cf_d(:,:,NROW)=Cf;
        NROW=NROW+1; 
    end
end
[Vcsr Icsr]=max(Result_PCA_DSM(:,end));
stats=statsOfMeasure(Cf_d(:,:,Icsr),1);
writetable(stats,'Dummy.xlsx','sheet',1);
writematrix(Result_PCA_DSM(Icsr,:),'Dummy.xlsx','sheet',2);

%% iii. KPCA - SM
clear st; clear ed; clear sttT; clear edtT;
st=3.5*fs+1:25:5*fs+1;
ed=5.5*fs:25:7*fs;
sttT=3.5*fs+1:10:5*fs+1;
edtT=5.5*fs:10:7*fs;
gamma=0.1:0.1:2;
NROW=1;
for epoch=1:length(st)
    st_time=floor(st(epoch));
    ed_time=floor(ed(epoch));

       for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPL_Temp=[];
            for c=1:Ch
                BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
            end
            BPL(:,trial)=BPL_Temp;
        end

        for trial=1:L
            Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
            BPR_Temp=[];
            for c=1:Ch
                BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
            end
            BPR(:,trial)=BPR_Temp;
        end
        clear Acc_g;
        for g=1:length(gamma)
            clear kpca; clear Rr; clear Rl; clear Ur; clear Ul; clear BPL_kpca; clear BPR_kpca;
            kpcaL = KernelPca(BPL', 'gaussian', 'gamma', gamma(g), 'AutoScale', false);
            kpcaR = KernelPca(BPR', 'gaussian', 'gamma', gamma(g), 'AutoScale', false);
            for d=1:dmax
                clear Acc_eptT;
                for epochtT=1:length(sttT)
                    sttT_time=floor(sttT(epochtT)); edtT_time=floor(edtT(epochtT));
                    clear Zpredict; clear BPTest;
                    for trial=1:NTest
                        clear Ytemp; clear RTt; clear UTt;clear BP_Temp; clear BPTt_Temp; 
                        Ytemp(:,:)=YTt(:,sttT_time:edtT_time,trial);
                        BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
                        BPTest=[];
                        for c=1:Ch
                            BPTest=[BPTest;BP_Temp(:,c)];
                        end
                        PL_KPCA=project(kpcaL, BPTest', d);
                        PR_KPCA=project(kpcaR, BPTest', d);
                        [VV,Zpredict(trial)]=max([norm(PL_KPCA),norm(PR_KPCA)]);
                    end
                    [Cf order]=confusionmat(ZTt,Zpredict);
                    Acc_eptT(epochtT)=(Cf(1,1)+Cf(2,2))/NTest; 
                    Cf_mat(:,:,epochtT)=Cf;
                end
                [Acc_g(g) Iend]=max(Acc_eptT);
                stats=statsOfMeasure(Cf_mat(:,:,Iend),1);
                Result_KPCA_SM(NROW,:)=[epoch d g Acc_g(g)]
                max(Result_KPCA_SM(:,end))
                NROW=NROW+1; 
            end
        end
    end  

%% iv. KPCA-DSM
clear st; clear ed; clear sttT; clear edtT;
st=3.5*fs+1:25:5*fs+1;
ed=5.5*fs:25:7*fs;
stTt=3.5*fs+1:10:5*fs+1;
edTt=5.5*fs:10:7*fs;

gamma=0.1:0.1:2;
kernel='gaussian';
NROW=1; 
for epoch=1:length(st)
    st_time=floor(st(epoch));
    ed_time=floor(ed(epoch));

    for trial=1:L
        Ytemp(:,:)=YTn(:,st_time:ed_time,indL(trial));
        BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
        BPL_Temp=[];
        for c=1:Ch
            BPL_Temp=[BPL_Temp;BP_Temp(:,c)];
        end
        BPL(:,trial)=BPL_Temp;
    end

    for trial=1:L
        Ytemp(:,:)=YTn(:,st_time:ed_time,indR(trial));
        BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
        BPR_Temp=[];
        for c=1:Ch
            BPR_Temp=[BPR_Temp;BP_Temp(:,c)];
        end
        BPR(:,trial)=BPR_Temp;
    end

    BPL_centered=BPL-mean(BPL,2);
    BPR_centered=BPR-mean(BPR,2);

    for N=1:dmax-1
        clear F_g;
        for g=1:length(gamma)
            clear KL; clear VL; clear DL; 
            clear ind;
            kernel_params.gamma=gamma(g);    
            KL = k_matrix(BPL_centered, BPL_centered, kernel, kernel_params);
            [VL, DL] = eig(KL);
            [DL, ind] = sort(diag(DL), 'descend');
            VL=VL(:,ind);

            clear KR; clear VR; clear DR; 
            clear ind;
            KR = k_matrix(BPR_centered, BPR_centered, kernel, kernel_params);
            [VR, DR] = eig(KR);
            [DR, ind] = sort(diag(DR), 'descend');
            VR=VR(:,ind);

            clear AL; clear AR; clear KLR; clear KRL; 
            AL=VL(:,1:N);
            AR=VR(:,1:N);  
            KLR = k_matrix(BPL_centered, BPR_centered, kernel, kernel_params);
            KRL = k_matrix(BPR_centered, BPL_centered, kernel, kernel_params);

            clear DL; clear DR; clear DLR; clear DRL; clear D; clear Btemp; clear d; 
            clear BP_centered; clear KL_pj; clear KR_pj; clear ind;
            DL=AL'*KL*AL;
            DR=AR'*KR*AR;
            DLR=AL'*KLR*AR;
            DRL=AR'*KRL*AL;
            D =[[DL DLR];[DRL DR]];
            [Btemp ltemp]=eig(D);
            [d ind]=sort(diag(ltemp));
            BP_centered=[BPL_centered BPR_centered];
            KL_pj = k_matrix(BP_centered, BPL_centered, kernel, kernel_params);
            KR_pj = k_matrix(BP_centered, BPR_centered, kernel, kernel_params);

            clear F_Nd;
            for Nd=2:2*N                % Dimension of difference subspace
                clear B; clear BPLpj_KDS; clear BPRpj_KDS; clear UL; clear UR;
                B=Btemp(:,ind(1:Nd));   
                BPLpj_KDS=B'*[[AL' zeros(N,L)];[zeros(N,L) AR']]*KL_pj;  
                BPRpj_KDS=B'*[[AL' zeros(N,L)];[zeros(N,L) AR']]*KR_pj;  
                F_Ds=FishScore(BPLpj_KDS,BPRpj_KDS);
                [UL,EL,VL]=svd(cov(BPLpj_KDS'));
                [UR,ER,VR]=svd(cov(BPRpj_KDS'));
                for dDs=1:Nd-1
                    clear Acc_ep;
                    for epochTt=1:length(stTt)
                        stTt_time=floor(stTt(epochTt));
                        edTt_time=floor(edTt(epochTt));
                        for trial=1:length(ZTt)
                            clear Ytemp; clear BPTt; clear KTt_pj; clear BPTtpj_KDS; clear UTt; clear ETt;
                            clear P1; clear P2;
                            Ytemp(:,:)=YTt(:,stTt_time:edTt_time,trial);
                            BP_Temp(:,:)=[bandpower(Ytemp',fs,[8 12]);bandpower(Ytemp',fs,[12 30])];
                            BPTt=[];
                            for c=1:Ch
                                BPTt=[BPTt;BP_Temp(:,c)];
                            end
                            KTt_pj = k_matrix(BP_centered, BPTt', kernel, kernel_params);
                            BPTtpj_KDS=B'*[[AL' zeros(N,L)];[zeros(N,L) AR']]*KTt_pj; 
                            P1=UL(:,1:dDs)'*BPTtpj_KDS;
                            P2=UR(:,1:dDs)'*BPTtpj_KDS;
                            [VV,Zpredict(trial)]=max([norm(P1),norm(P2)]);
                        end
                        clear Cf;
                        [Cf order]=confusionmat(ZTt,Zpredict);
                        Acc_ep(epochTt)=(Cf(1,1)+Cf(2,2))/length(ZTt);
                    end
                    Result_KPCA_DSM(NROW,:)=[epoch N g Nd dDs max(Acc_ep)]
                    max(Result_KPCA_DSM(:,end))
                    NROW=NROW+1;
                end         
            end 
        end    
    end
end
writematrix(Result_KPCA_DSM,'Dummy.xlsx')
[Vcsr Icsr]=max(Result_KPCA_DSM(:,end))
writematrix(Result_KPCA_DSM(Icsr,:),'Dummy.xlsx','sheet',2)

%% V. ISOMap-SM
clear st; clear ed; clear sttT; clear edtT;
st=3.5*fs+1:50:5*fs+1;
ed=5.5*fs:50:7*fs;
sttT=3.5*fs+1:25:5*fs+1;
edtT=5.5*fs:25:7*fs;
alpha=[0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.5];
eta=[0.001 0.01 0.1 0.5 1 10 100 1000];
gamma=[0.001 0.01 0.1 1 10 100 1000];
zeta=0.001; 

NROW=1;
for epoch=1:length(st)
    st_time=st(epoch);
    ed_time=ed(epoch);

    clear XTn;
    for trial=1:length(ZTn)
        clear YTemp; clear BP_Temp;
        YTemp(:,:)=YTn(:,st_time:ed_time,trial);
        BP_Temp(:,:)=[bandpower(YTemp',fs,[8 12]) bandpower(YTemp',fs,[12 30])];
        XTn(:,trial)=BP_Temp;
    end

    clear X2; clear dist; clear RNw; clear VNb; clear LCL;
    [M,N]=size(XTn);
    X2 = sum(XTn.^2,1);
    dist = abs(repmat(X2,N,1)+repmat(X2',1,N)-2*XTn'*XTn);   % ||Xi-Xj||.^2;
    [RNw,VNb,LCL] = SD_Isomap(dist,ZTn);

    for i_alpha=1:length(alpha)
        for i_eta=1:length(eta)
            for i_gamma=1:length(gamma)
                clear M2; clear M3; clear vec; clear val; clear ind;
                M2=XTn*((2*(1-alpha(i_alpha)).*RNw)+(alpha(i_alpha).*VNb)+(eta(i_eta).*LCL))*XTn';
                M3=inv(XTn*XTn'+zeta.*eye(M))*(M2-gamma(i_gamma).*eye(M));
                [vec, val] = eig(M3);
                [val, ind] = sort(real(diag(val)), 'descend'); 
                for m=2:6
                    clear YTn_Lowdim; clear Pj_vec;      
                    Pj_vec = vec(:,ind(1:m));
                    YTn_Lowdim = Pj_vec'*XTn;

                    clear R1; clear R2;
                    R1=cov(YTn_Lowdim(:,indL)');
                    R2=cov(YTn_Lowdim(:,indR)');

                    clear U1; clear E1; clear U2; clear E2;
                    [U1 E1 ~]=svd(R1); [U2 E2 ~]=svd(R2); 

                    % SM Classification 
                    clear Acc_d_SM;
                    for d_SM=1:m-1
                        clear Acc_ep; clear cf;
                        for epochTt=1:length(sttT)
                            clear Zpredict; 
                            for trial=1:length(ZTt)
                                clear YTemp; clear BP_Temp; clear YTemp; clear P1; clear P2; clear XTt;
                                YTemp(:,:)=YTt(:,sttT(epochTt):edtT(epochTt),trial);
                                XTt=[bandpower(YTemp',fs,[8 12]) bandpower(YTemp',fs,[12 30])]';
                                clear YTemp;
                                YTemp(:,:)=Pj_vec'*XTt;
                                P1=U1(:,1:d_SM)'*YTemp;
                                P2=U2(:,1:d_SM)'*YTemp;
                                [VV,Zpredict(trial)]=max([norm(P1),norm(P2)]);
                            end
                            clear cf;
                            cf=confusionchart(ZTt,Zpredict);
                            Acc_ep(epochTt)=sum(diag(cf.NormalizedValues))/sum(cf.NormalizedValues,"all");
                        end
                        Acc_d_SM(d_SM)=max(Acc_ep);         
                    end
                    Result_ISOMap_SM(NROW,:)=[epoch alpha(i_alpha) eta(i_eta) gamma(i_gamma) m max(Acc_d_SM)]
                    max(Result_ISOMap_SM(:,end))
                    NROW=NROW+1; 
                end
            end
        end
        writematrix(Result_ISOMap_SM,'Dummy_SM3_Update.xlsx');
        [Mcsr Icsr]=max(Result_ISOMap_SM(:,end));
        writematrix(Result_ISOMap_SM(Icsr,:),'Dummy_SM3_Update.xlsx','sheet',2);
    end    
    ACC_ISop_SM(epoch)=max(Result_ISOMap_SM(:,end))
end
 
%% Vi. ISOMap-DSM
epoch_DSM=Result_ISOMap_SM(Icsr,1);
alpha_DSM=Result_ISOMap_SM(Icsr,2);
eta_DSM=Result_ISOMap_SM(Icsr,3);
gamma_DSM=Result_ISOMap_SM(Icsr,4);

st_time=st(epoch_DSM);
ed_time=ed(epoch_DSM);

clear XTn;
for trial=1:length(ZTn)
    clear YTemp; clear BP_Temp;
    YTemp(:,:)=YTn(:,st_time:ed_time,trial);
    BP_Temp(:,:)=[bandpower(YTemp',fs,[8 12]) bandpower(YTemp',fs,[12 30])];
    XTn(:,trial)=BP_Temp;
end

clear X2; clear dist; clear RNw; clear VNb; clear LCL;
[M,N]=size(XTn);
X2 = sum(XTn.^2,1);
dist = abs(repmat(X2,N,1)+repmat(X2',1,N)-2*XTn'*XTn);   % ||Xi-Xj||.^2;
[RNw,VNb,LCL] = SD_Isomap(dist,ZTn);

clear M2; clear M3; clear vec; clear val; clear ind;
M2=XTn*((2*(1-alpha_DSM).*RNw)+(alpha_DSM.*VNb)+(eta_DSM.*LCL))*XTn';
M3=inv(XTn*XTn'+zeta.*eye(M))*(M2-gamma_DSM.*eye(M));
[vec, val] = eig(M3);
[val, ind] = sort(real(diag(val)), 'descend'); 

NROW_DSM=1;
for m=2:6
    clear YTn_Lowdim; clear Pj_vec;      
    Pj_vec = vec(:,ind(1:m));
    YTn_Lowdim = Pj_vec'*XTn;

    clear R1; clear R2;
    R1=cov(YTn_Lowdim(:,indC1)');
    R2=cov(YTn_Lowdim(:,indC2)');

    clear U1; clear E1; clear U2; clear E2;
    [U1 E1 ~]=svd(R1); [U2 E2 ~]=svd(R2); 

    P=0;Q=0;
    for d_DSM=1:m-1
        P=P+U1(:,d_DSM)*U1(:,d_DSM)';
        Q=Q+U2(:,d_DSM)*U2(:,d_DSM)';
    end

    clear UDS; clear EDS;
    [UDS,EDS,~]=svd(P+Q);

    clear D; clear YTn1_DS; clear YTn2_DS; clear R1DS; clear R2DS; clear U1DS; clear U2DS;
    D=UDS(:,find(diag(EDS<1)));
    dDs_max=length(find(diag(EDS<1)));
    YTn1_DS=D'*YTn_Lowdim(:,indL);
    YTn2_DS=D'*YTn_Lowdim(:,indR);
    R1DS=cov(YTn1_DS'); R2DS=cov(YTn2_DS'); 
    [U1DS E1DS ~]=svd(R1DS); [U2DS E2DS ~]=svd(R2DS);   
    for dDs=1:dDs_max
        clear Acc_ep_DSM;
        for epochTt=1:length(sttT)
            clear Zpredict; 
            for trial=1:length(ZTt)
                clear YTemp; clear BP_Temp; clear YTemp; clear P1; clear P2; clear XTt;
                YTemp(:,:)=YTt(:,sttT(epochTt):edtT(epochTt),trial);
                XTt=[bandpower(YTemp',fs,[8 12]) bandpower(YTemp',fs,[12 30])]';
                clear YTt_Lowdim;
                YTt_Lowdim(:,:)=Pj_vec'*XTt;
                clear YTtDS; clear P1; clear P2;
                YTtDS=D'*YTt_Lowdim;
                P1=U1DS(:,1:dDs)'*YTtDS;
                P2=U2DS(:,1:dDs)'*YTtDS;
                [VV,Zpredict(trial)]=max([norm(P1),norm(P2)]);
            end
            clear cf;
            cf=confusionchart(ZTt,Zpredict);
            Acc_ep_DSM(epochTt)=sum(diag(cf.NormalizedValues))/sum(cf.NormalizedValues,"all");
        end
        Result_DSM(NROW_DSM,:)=[m dDs max(Acc_ep_DSM)];
        max(Result_DSM(:,end))
        NROW_DSM=NROW_DSM+1;
    end
end
writematrix(Result_DSM,'Dummy_DSM.xlsx');
[Mcsr Icsr]=max(Result_DSM(:,end));
writematrix(Result_DSM(Icsr,:),'Dummy_DSM.xlsx','sheet',2);

