// @ts-ignore
function randomFromInterval(min:number, max:number) { // min and max included 
    return Math.random() * (max - min) + min
}
  
class Perceptron {
    ws: number[]
    learning_constant: number;
    constructor(n_inputs: number) {
        const ws:number[] = [];
        for (let i = 0; i < n_inputs; i++) {
            ws[i] = randomFromInterval(0, 1)
        }
        this.ws = ws;
        this.learning_constant = 0.01;
    }
    activate(input:number) {
        return input > 0 ? 1 : -1
    }
    feedforward(inputs: number[]) {
        let sum = 0;
        for (let i = 0; i < inputs.length; i++) {
            sum+= inputs[i]*this.ws[i]
        }
        return this.activate(sum)
    }
    train(inputs:number[], desired:number) {
        // Y
        const guess = this.feedforward(inputs);
        const error = desired - guess;
        for (let i = 0; i < inputs.length; i++) {
            this.ws[i] += this.learning_constant * error * inputs[i]
        }
    }
}

const f = (x:number):number => 2*x+1

class Trainer {
    inputs: number[]
    answer: number
    constructor(x: number, y:number, a: number) {
        this.answer = a
        this.inputs = [x, y, 1]
    }
}



function setup_and_train(n_training: number, ptron: Perceptron) {
    console.log('Init setup')
    const trainers:Trainer[] = []
    for (let i = 0; i < n_training; i++) {
        const x = randomFromInterval(-100, 100)
        const y = randomFromInterval(-100, 100)
        const a = y < f(x) ? -1 : 1
        trainers.push(new Trainer(x, y , a))
    }
    console.log('trainers length', trainers.length)
    console.log('Init train')
    
    for (let i = 0; i < n_training; i++) {
        ptron.train(trainers[i].inputs, trainers[i].answer)
    }
    console.log(ptron.ws)
}
let falso_negativos = 0;
let falso_positivos = 0;
function test(n_test: number, ptron: Perceptron) {
    let errors = 0;
    for (let i = 0; i < n_test; i++) {
        const x = randomFromInterval(-100, 100)
        const y = randomFromInterval(-100, 100)
        const a = y < f(x) ? -1 : 1
        const guess = ptron.feedforward([x, y, 1])
        if (guess != a) {
            a === -1? falso_positivos++ : falso_negativos++
            errors++
        }
    }
    // 1 = bom, 0 = ruim
    const e = (n_test-errors)/n_test
    console.log({errors, e, falso_positivos, falso_negativos})
    return e
}

console.clear()
const ptron = new Perceptron(3)
setup_and_train(2000, ptron)
test(100, ptron)
