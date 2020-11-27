package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/labstack/echo"
)

type Data struct {
	Pregnancies   float64 `json:"Pregnancies"`
	Glucose       float64 `json:"Glucose"`
	BloodPressure float64 `json:"BloodPressure"`
	SkinThickness float64 `json:"SkinThickness"`
	Insulin       float64 `json:"Insulin"`
	BMI           float64 `json:"BMI"`
	DPF           float64 `json:"DPF"`
	Age           float64 `json:"Age"`
	Outcome       float64 `json:"Outcome"`
}

var columnsNamesNN []string
var columnsNN map[string]int
var dataframeNN [][]float64

var dataTest [][]float64
var dataTrain [][]float64

var xDataTest [][]float64
var xDataTrain [][]float64

var yDataTest [][]float64
var yDataTrain [][]float64

var predictedResults [][]float64
var confusionMatrix [][]int

var data Data

func readArchiveCSV(url string) ([]string, map[string]int, [][]float64) {
	resp, err := http.Get(url)

	if err != nil {
		log.Fatal("No se puede leer el archivo de entrada ", err)
	}
	defer resp.Body.Close()
	csvReader := csv.NewReader(resp.Body)

	fileData, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("No se puede parsear el archivo de entrada ", err)
	}

	headers := make([]string, len(fileData[0]))
	copy(headers, fileData[0])

	columns := make(map[string]int)
	for i, header := range headers {
		columns[header] = i
	}

	fileData = fileData[1:]
	fileDataReal := make([][]float64, len(fileData))

	for i := range fileDataReal {
		fileDataReal[i] = make([]float64, len(headers))
		for j := range fileDataReal[i] {
			val, _ := strconv.ParseFloat(fileData[i][j], 64)
			fileDataReal[i][j] = float64(val)
		}
	}

	return headers, columns, fileDataReal
}

func splitPercent(fileData [][]float64, percentSplit float64) ([][]float64, [][]float64) {
	newfileData1 := make([][]float64, 0)
	newfileData2 := make([][]float64, 0)

	for i := 0; i < len(fileData); i++ {
		s1 := rand.NewSource(time.Now().UnixNano())
		r1 := rand.New(s1)
		if percentSplit < r1.Float64() {
			newfileData1 = append(newfileData1, fileData[i])
		} else {
			newfileData2 = append(newfileData2, fileData[i])
		}
	}
	return newfileData1, newfileData2
}

func splitColumns(headers []string, columns map[string]int, fileData [][]float64, newheaders []string) ([]string, map[string]int, [][]float64) {

	temp := make([]int, len(newheaders))
	newfileData := make([][]float64, len(fileData))

	for i, newh := range newheaders {
		temp[i] = columns[newh]
	}

	for i := range newfileData {
		newfileData[i] = make([]float64, len(temp))
		for j, t := range temp {
			newfileData[i][j] = fileData[i][t]
		}
	}

	newcolumns := make(map[string]int)

	for i, header := range newheaders {
		newcolumns[header] = i
	}
	return newheaders, newcolumns, newfileData
}

func GetMinMax(array []float64) (float64, float64) {
	var max float64 = float64(0)
	var min float64 = float64(0)
	for _, value := range array {
		if max < value {
			max = value
		}
		if min > value {
			min = value
		}
	}
	return min, max
}

func GetCol(arr [][]float64, colID int) []float64 {
	out := []float64{}
	for _, row := range arr {
		out = append(out, row[colID])
	}
	return out
}

func normalizeData(fileData [][]float64) [][]float64 {
	newfileData := make([][]float64, len(fileData))
	for i := 0; i < len(fileData); i++ {
		newfileData[i] = make([]float64, len(fileData[i]))
	}

	min := make([]float64, len(fileData[0]))
	max := make([]float64, len(fileData[0]))

	for i := 0; i < len(fileData[0]); i++ {
		min[i], max[i] = GetMinMax(GetCol(fileData, i))
		for j := 0; j < len(fileData); j++ {
			newfileData[j][i] = (fileData[j][i] - min[i]) / (max[i] - min[i])
		}
	}

	return newfileData
}

type NeuralNetwork struct {
	mHiddenLayer      []*Neural
	mInputLayer       []*Neural
	mOutputLayer      []*Neural
	mWeightHidden     [][]float64
	mWeightOutput     [][]float64
	mLastChangeHidden [][]float64
	mLastChangeOutput [][]float64
	mOutput           []float64
	mForwardDone      chan bool
	mFeedbackDone     chan bool
	mRegression       bool
	mRate1            float64 //learning rate
	mRate2            float64
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func makeMatrix(rows, colums int, value float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = value
		}
	}
	return mat
}

func randomMatrix(rows, colums int, lower, upper float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = rand.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

func DefaultNetwork(iInputCount, iHiddenCount, iOutputCount int, iRegression bool) *NeuralNetwork {
	return NewNetwork(iInputCount, iHiddenCount, iOutputCount, iRegression, 0.01, 0.001)
}

func NewNetwork(iInputCount, iHiddenCount, iOutputCount int, iRegression bool, iRate1, iRate2 float64) *NeuralNetwork {
	iInputCount += 1
	network := &NeuralNetwork{}
	network.mRegression = iRegression
	network.mOutput = make([]float64, iOutputCount)
	network.mForwardDone = make(chan bool)
	network.mFeedbackDone = make(chan bool)
	network.mInputLayer = make([]*Neural, iInputCount)
	network.mRate1 = iRate1
	network.mRate2 = iRate2
	for i := 0; i < iInputCount; i++ {
		network.mInputLayer[i] = NewNeural(network, 0, i, 1)
	}
	network.mHiddenLayer = make([]*Neural, iHiddenCount)
	for i := 0; i < iHiddenCount; i++ {
		network.mHiddenLayer[i] = NewNeural(network, 1, i, iInputCount)
	}
	network.mOutputLayer = make([]*Neural, iOutputCount)
	for i := 0; i < iOutputCount; i++ {
		network.mOutputLayer[i] = NewNeural(network, 2, i, iHiddenCount)
	}

	network.mWeightHidden = randomMatrix(iInputCount, iHiddenCount, -0.2, 0.2)
	network.mWeightOutput = randomMatrix(iHiddenCount, iOutputCount, -2.0, 2.0)

	network.mLastChangeHidden = makeMatrix(iInputCount, iHiddenCount, 0.0)
	network.mLastChangeOutput = makeMatrix(iHiddenCount, iOutputCount, 0.0)

	return network
}

func (self *NeuralNetwork) Start() { //start all the neurals in the network
	for _, n := range self.mInputLayer {
		n.start(self.mRegression)
	}
	for _, n := range self.mHiddenLayer {
		n.start(self.mRegression)
	}
	for _, n := range self.mOutputLayer {
		n.start(self.mRegression)
	}
}

func (self *NeuralNetwork) Stop() { //start all the neurals in the network

	for _, n := range self.mInputLayer {
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	for _, n := range self.mHiddenLayer {
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	for _, n := range self.mOutputLayer {
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	close(self.mForwardDone)
	close(self.mFeedbackDone)
}

func (self *NeuralNetwork) Forward(input []float64) (output []float64) {
	if len(input)+1 != len(self.mInputLayer) {
		panic("amount of input variable doesn't match")
	}
	go func() {
		for i := 0; i < len(self.mInputLayer)-1; i++ {
			self.mInputLayer[i].mInputChan <- input[i]
		}
		self.mInputLayer[len(self.mInputLayer)-1].mInputChan <- 1.0 //bias node
	}()
	for i := 0; i < len(self.mOutput); i++ {
		<-self.mForwardDone
	}
	return self.mOutput[:]
}

func (self *NeuralNetwork) Feedback(target []float64) {
	go func() {
		for i := 0; i < len(self.mOutput); i++ {
			self.mOutputLayer[i].mFeedbackChan <- target[i]
		}
	}()
	for i := 0; i < len(self.mHiddenLayer); i++ {
		<-self.mFeedbackDone
	}

}

func (self *NeuralNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(self.mOutput); i++ {
		err := self.mOutput[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(N int) []int {
	A := make([]int, N)
	for i := 0; i < N; i++ {
		A[i] = i
	}
	//randomize
	for i := 0; i < N; i++ {
		j := i + int(rand.Float64()*float64(N-i))
		A[i], A[j] = A[j], A[i]
	}
	return A
}

func (self *NeuralNetwork) Train(start, fin int, inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0])+1 != len(self.mInputLayer) {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.mOutputLayer) {
		panic("amount of output variable doesn't match")
	}
	old_err1 := 1.0
	old_err2 := 2.0

	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		for j := start; j < fin; j++ {
			self.Forward(inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
		}
		if i%100 == 0 {
			last_target := targets[len(targets)-1]
			cur_err := self.CalcError(last_target)
			fmt.Println("err: ", cur_err, "i:", i)
			if (old_err2-old_err1 < 0.001) && (old_err1-cur_err < 0.001) { //early stop
				break
			}
			old_err2 = old_err1
			old_err1 = cur_err

		}
	}
}

func (self *NeuralNetwork) ActiveFunction(a []float64) []float64 {
	var classValue []float64
	for i := range a {
		if a[i] > 0.5 {
			classValue = append(classValue, 1)
		} else {
			classValue = append(classValue, 0)
		}
	}
	return classValue
}

func (self *NeuralNetwork) Test(patternsX [][]float64, patternsY [][]float64, print bool) [][]float64 {
	predictedArray := make([][]float64, 0)
	for i := range patternsX {
		output := self.Forward(patternsX[i])
		foo := make([]float64, 0)
		for j := range output {
			calc := sigmoid(output[j])
			foo = append(foo, calc)
			if print {
				fmt.Println(patternsX[i], "->\t\t", calc, "->\tClase Predicha", self.ActiveFunction(foo), ":", patternsY[i])
			}
		}
		predictedArray = append(predictedArray, foo)
	}
	return predictedArray
}

type Neural struct {
	mInputChan    chan float64
	mFeedbackChan chan float64
	mInputCount   int
	mLayer        int
	mNo           int
	mNetwork      *NeuralNetwork
	mValue        float64
}

func NewNeural(iNetwork *NeuralNetwork, iLayer, iNo, iInputCount int) *Neural {
	nerual := &Neural{}
	nerual.mNetwork = iNetwork
	nerual.mInputCount = iInputCount
	nerual.mLayer = iLayer
	nerual.mInputChan = make(chan float64)
	nerual.mFeedbackChan = make(chan float64)
	nerual.mNo = iNo
	nerual.mValue = 0.0
	return nerual
}

func (self *Neural) start(regression bool) {
	go func() { //forward loop
		defer func() { recover() }()
		for {
			sum := 0.0
			for i := 0; i < self.mInputCount; i++ {
				value := <-self.mInputChan
				sum += value
			}
			if self.mLayer == 0 { //input layer
				for i := 0; i < len(self.mNetwork.mHiddenLayer); i++ {
					self.mNetwork.mHiddenLayer[i].mInputChan <- sum * self.mNetwork.mWeightHidden[self.mNo][i]
				}
			} else if self.mLayer == 1 { //hidden layer
				sum = sigmoid(sum)
				for i := 0; i < len(self.mNetwork.mOutputLayer); i++ {
					self.mNetwork.mOutputLayer[i].mInputChan <- sum * self.mNetwork.mWeightOutput[self.mNo][i]
				}
			} else { //output layer
				if !regression {
					sum = sigmoid(sum)
				}
				self.mNetwork.mOutput[self.mNo] = sum
				self.mNetwork.mForwardDone <- true
			}
			self.mValue = sum
		}

	}()

	go func() { //feedback loop
		defer func() { recover() }()
		for {
			if self.mLayer == 0 { //input layer
				return
			} else if self.mLayer == 1 { //hidden layer
				err := 0.0
				for i := 0; i < len(self.mNetwork.mOutput); i++ {
					err += <-self.mFeedbackChan
				}
				for i := 0; i < self.mInputCount; i++ {
					change := err * dsigmoid(self.mValue) * self.mNetwork.mInputLayer[i].mValue
					self.mNetwork.mWeightHidden[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeHidden[i][self.mNo])
					self.mNetwork.mLastChangeHidden[i][self.mNo] = change
				}
				self.mNetwork.mFeedbackDone <- true
			} else { //output layer
				target := <-self.mFeedbackChan
				err := self.mValue - target
				for i := 0; i < self.mInputCount; i++ {
					self.mNetwork.mHiddenLayer[i].mFeedbackChan <- err * self.mNetwork.mWeightOutput[i][self.mNo]
				}
				if regression {
					for i := 0; i < self.mInputCount; i++ {
						change := err * self.mNetwork.mHiddenLayer[i].mValue
						self.mNetwork.mWeightOutput[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeOutput[i][self.mNo])
						self.mNetwork.mLastChangeOutput[i][self.mNo] = change
					}
				} else {
					for i := 0; i < self.mInputCount; i++ {
						change := err * dsigmoid(self.mValue) * self.mNetwork.mHiddenLayer[i].mValue
						self.mNetwork.mWeightOutput[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeOutput[i][self.mNo])
						self.mNetwork.mLastChangeOutput[i][self.mNo] = change
					}
				}

			}
		}
	}()
}

func GetConfusionMatrix(predicted [][]float64, real [][]float64) [][]int {
	min, max := GetMinMax(GetCol(real, 0))
	n := max - min + 1
	np, nr := int(n), int(n)
	cm := make([][]int, nr)
	for i := 0; i < nr; i++ {
		cm[i] = make([]int, np)
	}
	for i := range real {
		for j := range real[0] {
			a := int(real[i][j])
			b := int(predicted[i][j])
			cm[a][b] = cm[a][b] + 1
		}
	}
	return cm
}

func PrintConfusionMatrix(matrix [][]int) {
	fmt.Println("Confusion Matrix")
	fmt.Print("   ")
	for j := len(matrix[0]) - 1; j >= 0; j-- {
		fmt.Print("|", j, "|")
	}
	fmt.Println()
	for i := len(matrix) - 1; i >= 0; i-- {
		fmt.Print("|", i, "|")
		for j := len(matrix[0]) - 1; j >= 0; j-- {
			fmt.Print("|", matrix[i][j])
		}
		fmt.Println("|")
	}
}

func Recall(matrix [][]int) float64 {
	return float64(matrix[1][1]) / float64(matrix[1][1]+matrix[0][1])
}

func Precision(matrix [][]int) float64 {
	return float64(matrix[1][1]) / float64(matrix[1][1]+matrix[1][0])
}
func Accuracy(matrix [][]int) float64 {
	return float64(matrix[1][1]+matrix[0][0]) / float64(matrix[0][0]+matrix[1][0]+matrix[0][1]+matrix[1][1])
}
func Metrics(matrix [][]int) {
	fmt.Println("Metricas")
	fmt.Printf("\tRecall: %.2f ", Recall(matrix)*100)
	fmt.Println()
	fmt.Printf("\tPrecision: %.2f", Precision(matrix)*100)
	fmt.Println()
	fmt.Printf("\tAccuracy: %.2f", Accuracy(matrix)*100)
}

func getData(c echo.Context) (err error) {
	data := new(Data)
	if err = c.Bind(data); err != nil {
		return
	}
	return c.JSON(http.StatusOK, data)
}

func FloatToString(input_num []float64) string {
	// to convert a float number to a string
	var phrase string = ""
	for i := 0; i < len(input_num); i++ {
		phrase += " " + strconv.FormatFloat(input_num[i], 'f', 6, 64)
	}
	return phrase
}

func main() {
	split := 0.8
	columnsNamesNN, columnsNN, dataframeNN = readArchiveCSV("https://raw.githubusercontent.com/jysique/datasets/master/Dataset/diabetes.csv")
	dataframeNN = normalizeData(dataframeNN)
	dataTest, dataTrain = splitPercent(dataframeNN, split)
	_, _, xDataTrain = splitColumns(columnsNamesNN, columnsNN, dataTrain, columnsNamesNN[:len(columnsNamesNN)-1])
	_, _, yDataTrain = splitColumns(columnsNamesNN, columnsNN, dataTrain, []string{columnsNamesNN[len(columnsNamesNN)-1]})
	_, _, xDataTest = splitColumns(columnsNamesNN, columnsNN, dataTest, columnsNamesNN[:len(columnsNamesNN)-1])
	_, _, yDataTest = splitColumns(columnsNamesNN, columnsNN, dataTest, []string{columnsNamesNN[len(columnsNamesNN)-1]})

	inputs := len(xDataTrain[0])

	nn := DefaultNetwork(inputs, int(inputs/2), 1, false)
	nn.Start()
	//for i := 0; i < len(xDataTrain)/2; i++ {
	//	fmt.Println(xDataTrain[i])

	//}
	//nn.Train(xDataTrain, yDataTrain, 1000)

	//predictedResults := nn.Test(xDataTest, yDataTest, false)

	//confusionMatrix = GetConfusionMatrix(predictedResults, yDataTest)
	//PrintConfusionMatrix(confusionMatrix)
	//Metrics(confusionMatrix)
	//nn.Stop()
	var option string
	// var train, retreat int
	local := os.Args[1]
	remotes := os.Args[2:]
	ch := make(chan Msg)
	go server(local, remotes, xDataTrain, yDataTrain, *nn, ch)
	fmt.Print("Your option:")
	fmt.Scanf("%s", &option)
	sendAll(option, local, remotes, *nn, xDataTest, yDataTest)

	// if option == "train" {
	// 	train = 1
	// } else {
	// 	retreat = 1
	// }
	for range remotes {
		msg := <-ch
		if msg.Option == "test" {
			fmt.Println("Test")
		} else {
			// retreat++
		}
	}
	// if train > retreat {
	// 	fmt.Println("TRAIN!! ")
	// } else {
	// 	fmt.Println("run...")
	// }
}

func server(local string, remotes []string, _Xdatatrain, _Ydatatrain [][]float64, _nn NeuralNetwork, ch chan Msg) {
	if ln, err := net.Listen("tcp", local); err == nil {
		defer ln.Close()
		fmt.Printf("Listening on %s\n", local)
		for {
			if conn, err := ln.Accept(); err == nil {
				fmt.Printf("Connection accepted from %s\n", conn.RemoteAddr())
				go handle(conn, local, remotes, _Xdatatrain, _Ydatatrain, _nn, ch)
			}
		}
	}
}

// Msg lorem ipsum
type Msg struct {
	Addr   string `json:"addr"`
	Option string `json:"option"`
}

func printData(start, finish int, _datatrain [][]float64) {
	for i := start; i < finish; i++ {
		fmt.Println(_datatrain[i])
	}
}

func handle(conn net.Conn, local string, remotes []string, _Xdatatrain, _Ydatatrain [][]float64, _nn NeuralNetwork, ch chan Msg) {
	defer conn.Close()
	dec := json.NewDecoder(conn)
	var msg Msg
	if err := dec.Decode(&msg); err == nil {
		switch msg.Option {
		case "train":
			//fmt.Printf("Message: %v\n", msg)
			xsize := len(_Xdatatrain) / 5

			res1 := strings.SplitAfterN(local, "localhost:800", 2)
			res, _ := strconv.Atoi(res1[1])
			//fmt.Println("comienza en: ", (res)*xsize, " termina en: ", (res+1)*xsize)
			if res < 4 {
				_nn.Train((res)*xsize, (res+1)*xsize, _Xdatatrain, _Ydatatrain, 1000)
			} else {
				_nn.Train((res)*xsize, len(_Xdatatrain)-1, _Xdatatrain, _Ydatatrain, 1000)
			}
			//	printData((res)*xsize, (res+1)*xsize, _datatrain)
			//sendAll("test", local, remotes)
			ch <- msg
		case "test":
			fmt.Println("Termine 1")
		case "server":
			e := echo.New()
			e.GET("/prediction", func(c echo.Context) error {
				array := [][]float64{{data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI, data.DPF, data.Age}}
				array2 := [][]float64{{data.Outcome}}
				predictedResults2 := _nn.Test(array, array2, false)
				return c.String(http.StatusOK, "La clase de la instancia es: "+FloatToString(_nn.ActiveFunction(predictedResults2[0])))
			})
			e.POST("/data", getData)

			e.Start(":8080")
		}
	}
}

func sendAll(option, local string, remotes []string, _nn NeuralNetwork, _Xdatatest, _Ydatatest [][]float64) {
	for _, remote := range remotes {
		send(local, remote, option, _nn, _Xdatatest, _Ydatatest)
	}
}

func send(local, remote, option string, _nn NeuralNetwork, _Xdatatest, _Ydatatest [][]float64) {

	if conn, err := net.Dial("tcp", remote); err == nil {
		enc := json.NewEncoder(conn)
		if err := enc.Encode(Msg{local, option}); err == nil {
			switch option {
			case "test":
				predictedResults := _nn.Test(_Xdatatest, _Ydatatest, false)
				confusionMatrix = GetConfusionMatrix(predictedResults, _Ydatatest)
				PrintConfusionMatrix(confusionMatrix)
				Metrics(confusionMatrix)
				//_nn.Stop()
				fmt.Println("Termine 2")
			}
			fmt.Printf("Sending %s to %s\n", option, remote)
		}
	}
}
