clear all;
load("9094model.mat",'net');
%load("nbmodelv1.mat");

prompt = 'What is the text of the article you would like to classify?';
inputarticle = inputdlg(prompt);

% nbinput = cell2table(inputarticle);
% nbinput = renamevars(nbinput,"inputarticle","Text");
% 
% nblabel = predict(nbmodel,nbinput);
% nbfinallabel = char(nblabel);

inputarticle =preprocessText(inputarticle);




sequenceLength = 800;
enc = wordEncoding(inputarticle);
XNew = doc2sequence(enc,inputarticle,'Length',sequenceLength);

labelsNew = classify(net,XNew);

theLabel = char(labelsNew);
msgboxmessage = "Your article was predicted as the following category: ";
finalmessage = append(msgboxmessage,theLabel);
msgbox(finalmessage);