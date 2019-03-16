var config={
	delimiter: "",	// auto-detect
	newline: "",	// auto-detect
	quoteChar: '"',
	escapeChar: '"',
	header: false,
	transformHeader: undefined,
	dynamicTyping: false,
	preview: 0,
	encoding: "",
	worker: false,
	comments: false,
	step: undefined,
	complete: parsingComplete,
	error: undefined,
	download: true,
	skipEmptyLines: false,
	chunk: undefined,
	fastMode: undefined,
	beforeFirstChunk: undefined,
	withCredentials: undefined,
	transform: undefined,
	delimitersToGuess: [',', '\t', '|', ';', Papa.RECORD_SEP, Papa.UNIT_SEP]
};
var model;
var trainingdatax=[];
var trainingdatay=[];
var testdatax=[];
var testdatay=[];
var dataset;
function parsingComplete(result,file){
    dataset=result.data;
	// train();
	test();
}
function readCSV(){
    Papa.parse("Book.csv",config);   
}
async function train(){
    console.log(dataset.length);
    for(var i=1;i<8500;i++){
        var row=[];
        row[0]=parseInt(dataset[i][0])/40;
        row[1]=parseInt(dataset[i][1])/80;
        row[2]=parseInt(dataset[i][3])/600;
        trainingdatax.push(row);
        if(dataset[i][4]==="critical"){
			trainingdatay.push([1,0,0,0]);
		}
		else if(dataset[i][4]==="severe"){
			trainingdatay.push([0,1,0,0]);
		}
		else if(dataset[i][4]==="mild"){
			trainingdatay.push([0,0,1,0]);
		}
		else if(dataset[i][4]==="no"){
			trainingdatay.push([0,0,0,1]);
		}
	};
    model=tf.sequential();
    model.add(tf.layers.dense({units:6,activation:"sigmoid",inputShape:[3]}));
    model.add(tf.layers.dense({units:6,activation:"sigmoid"}));
    model.add(tf.layers.dense({units:4,activation:"sigmoid"}));
    model.compile({loss:"meanSquaredError",optimizer:"rmsprop"});
    var trainx=tf.tensor2d(trainingdatax);
    var trainy=tf.tensor2d(trainingdatay);
    await model.fit(trainx,trainy,{epochs:1000,shuffle:true,callbacks:{onEpochEnd:(epochs,logs)=>{
        console.log("Epoch ="+epochs+" Loss="+logs.loss);
    }}});
    console.log("Training Completed");
    model.save('downloads://my-model');
}
async function loadModel(){
	model=await tf.loadLayersModel('my-model.json');
	console.log("Model loaded");
	readCSV();
}
function predict(){
	var temp=document.getElementById("temp").value;
	var humidity=document.getElementById("humidity").value;
	var smoke=document.getElementById("smoke").value;
	var data=[];
	data[0]=parseInt(temp)/40;
	data[1]=parseInt(humidity)/80;
	data[2]=parseInt(smoke)/600;
	var x=tf.tensor2d([data]);
	var result=model.predict(x);
	result.print();
	var indextensor=result.argMax(1);
	var index=indextensor.dataSync()[0];
	var indexLabel=["critical","severe","mild","no"];
	document.getElementById("result").innerHTML=indexLabel[index];
	return indexLabel[index];
}
function test(){
	var totalCouunt=0;
	var correct=0;
	for(var i=8500;i<dataset.length;i++){
		totalCouunt++;
		var row=[];
        row[0]=parseInt(dataset[i][0])/40;
        row[1]=parseInt(dataset[i][1])/80;
		row[2]=parseInt(dataset[i][3])/600;
		var x=tf.tensor2d([row]);
		var result=model.predict(x);
		var indextensor=result.argMax(1);
		var index=indextensor.dataSync()[0];
		var indexLabel=["critical","severe","mild","no"];
		if(indexLabel[index]===dataset[i][4]){
			correct++;
		}

	}
	console.log("Accuracy= "+((correct/totalCouunt)*100));
}
loadModel();

