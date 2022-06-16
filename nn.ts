const links_ws: {
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

class OutputNeuron extends Neuron {

}
// add backpropagation ou outra forma de treinar
function nn() {

    const input_layer:InputNeuron[] = [];
    const bias = new InputNeuron(0)
    input_layer.push(bias)
    bias.in_value = 1;
    const x_input = createInputNeuron()
    x_input.in_value = 2;
    const y_input = createInputNeuron()
    y_input.in_value = 4;
    input_layer.push(x_input, y_input)
    
    const hidden_layer:Neuron[] = [];
    hidden_layer.push(createNeuron(), createNeuron(), createNeuron(), createNeuron())
    
    for (const unit of hidden_layer) {
        for (const input of input_layer) {
            createLink(input, unit)
        }
    }

    const output_layer:OutputNeuron[] = [createOutputNeuron(), createOutputNeuron()];

    for (const unit of output_layer) {
        for (const input of hidden_layer) {
            createLink(input, unit)
        }
    }

    // execute
    const hidden_response:NeuronOutput[] = []
    for (const unit of hidden_layer) {
        hidden_response.push(unit.receive(input_layer.map(unit => ({ origin: unit, value: unit.in_value }) )))
    }
    
    const output_response:NeuronOutput[] = [];
    for (const unit of output_layer) {
        output_response.push(unit.receive(hidden_response))
    }
    console.log(output_response.map(out => out.value))
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
function randomFromInterval(min:number, max:number) { // min and max included 
    return Math.random() * (max - min) + min
}
  
nn()