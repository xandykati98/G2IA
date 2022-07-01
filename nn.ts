let links_ws: {
    [link_name:string]: number
} = {

}
let last_id = 0;
// função ideal y=x^2
// treinar com limites de x(-10, 10) e y (0, 80)

type NeuronOutput = {
    origin: Neuron;
    value: number;
}

function createLink(origin: Neuron, end: Neuron) {
    links_ws[origin.id+'_to_'+end.id] = randomFromInterval(0, 1)
    return links_ws[origin.id+'_to_'+end.id]
}

function findLinkWeight(origin: Neuron, end: Neuron) {
    return links_ws[origin.id+'_to_'+end.id]
}
function findLink(origin: Neuron, end: Neuron) {
    return origin.id+'_to_'+end.id
}

class Neuron {
    id: number;
    constructor(id:number) {
        this.id = id;
    }
    receive(inputs: NeuronOutput[]) {
        const weighted_sum = this.input_function(inputs)
        
        return {
            origin: this,
            value: this.activation_function(weighted_sum)
        }
    }
    input_function(inputs: NeuronOutput[]): number {
        let sum = 0;
        for (const { value, origin } of inputs) {
            // Each unit j first computes a weighted sum of its inputs:
            sum += value * findLinkWeight(origin, this)
        }
        return sum
    }
    φ = this.activation_function;
    φ_derivative(input: number): number {
        return Math.exp(-input) / Math.pow(1 + Math.exp(-input), 2)
    }
    φ_derivative_alt(input: number): number {
        return this.φ(input) * (1 - this.φ(input))
    }
    activation_function(input: number): number {
        return 1.0/(1.0 + Math.exp(-input))
    }
}

class InputNeuron extends Neuron {
    in_value: number
    constructor(id:number) {
        super(id)
        this.in_value = 1
    }
}
const junct = <A, B>(arr1: A[], arr2:B[]):Array<[A,B, number]> => {
    return arr1.map((e, i) => [e, arr2[i], i])
}
class OutputNeuron extends Neuron {

}
const sum = (arr:number[]) => arr.reduce((a, b)=>a+b, 0)
// add backpropagation ou outra forma de treinar
function nn() {


    let input = [2, 1]
    let desired = [(input[1] > (input[0]**2)) ? 1 : 0] 
    const num_iteracoes = 100;
    const ŋ = 0.01;
    let es:number[] = []
    // estrutura

    const input_layer:InputNeuron[] = [];
    const bias = new InputNeuron(0)
    input_layer.push(bias)
    bias.in_value = 1;
    const x_input = createInputNeuron()
    x_input.in_value = input[0];
    const y_input = createInputNeuron()
    y_input.in_value = input[1];
    input_layer.push(x_input, y_input)
    
    const hidden_layer:Neuron[] = [];
    hidden_layer.push(createNeuron(), createNeuron(), createNeuron(), createNeuron())
    
    for (const unit of hidden_layer) {
        for (const input of input_layer) {
            createLink(input, unit)
        }
    }

    const output_layer:OutputNeuron[] = [createOutputNeuron()];

    for (const unit of output_layer) {
        for (const input of hidden_layer) {
            createLink(input, unit)
        }
    }

    let xxxxx = 0;
    while (xxxxx < num_iteracoes) {

    // execute
    const hidden_response:NeuronOutput[] = []
    for (const unit of hidden_layer) {
        hidden_response.push(unit.receive(input_layer.map(unit => ({ origin: unit, value: unit.in_value }) )))
    }
    
    const output_response:NeuronOutput[] = [];
    for (const unit of output_layer) {
        output_response.push(unit.receive(hidden_response))
    }
    /**
     * E = Erro global instantâneo (nessa iteração)
     * E(n) = 1/2  \sum_{j=1}^{J} e^2_{j}(n)
     */
    let E = (1/2)*sum(output_response.map((unit, j) => (desired[j] - unit.value)**2))

    console.log({E})
    // Vou pular o erro global médio pois estou usando iterações de 1 em 1

    /**
     * Calculo de gradiente local dos neuronios na camada de saida
     * δ_{j}(n)=-e_{j}(n)φ'_{j}(v_{j}(n))
     * δ_{j}(n) = -(desired[j] - unit.value) * unit.φ'(unit.value)
     */
    
    const δ_saida:{ origin: Neuron, value: number }[] = []
    for (const [neuron, output, j] of junct(output_layer, output_response)) {
        const δ = (desired[j] - output.value) * neuron.φ_derivative(output.value)
        δ_saida.push({ origin: neuron, value: δ })
    }

    /**
     * Calculo de delta de pesos entre a camada oculta e a camada de saida
     * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
     */
    
    const Δw_hidden_to_saida:number[] = []

    // output_hidden = y_i(n)
    // Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
    for (const output_hidden of hidden_response) {
        for (const { origin, value: δ } of δ_saida) {
            const Δw = -ŋ * δ * output_hidden.value
            Δw_hidden_to_saida.push(Δw)
            // update ws
            links_ws[findLink(output_hidden.origin, origin)] -= Δw
        }
    }


    es.push(E)
        xxxxx++
    }
    console.log(es)
}

// cria um neuron generico, o bias neuron deve ser criado a mão pela classe Neuron
function createNeuron() {
    last_id++
    return new Neuron(last_id)
}

function createInputNeuron() {
    last_id++
    return new InputNeuron(last_id)
}
function createOutputNeuron() {
    last_id++
    return new OutputNeuron(last_id)
}
// @ts-ignore
function randomFromInterval(min:number, max:number) { // min and max included 
    return Math.random() * (max - min) + min
}
  
nn()