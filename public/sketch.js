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
    train();
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
async function predict(){
    model=await tf.loadLayersModel('my-model.json');
    var x=tf.tensor2d([[29/40,75/80,538/600]]);
    var x1=tf.tensor2d([[40/40,75/80,252/600]]);
    model.predict(x1).print()

}
// readCSV();
predict();
