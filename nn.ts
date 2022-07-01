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

const chunkArray = <T>(arr:T[], size:number) => {
    const chunks:T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
        chunks.push(arr.slice(i, i + size));
    }
    return chunks;
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
    /**
     * Retorna um objeto contendo o neuronio e o valor da saída dele, ou, `y`
     */
    receive(inputs: NeuronOutput[]) {
        const weighted_sum = this.input_function(inputs)
        
        return {
            origin: this,
            value: this.activation_function(weighted_sum)
        }
    }
    /**
     * Computa os pesos e retorna o valor que vai ser passado para a função de ativação
     */
    input_function(inputs: NeuronOutput[]): number {
        let sum = 0;
        for (const { value, origin } of inputs) {
            // Each unit j first computes a weighted sum of its inputs:
            sum += value * findLinkWeight(origin, this)
        }
        return sum
    }
    /**
     * Função de ativação sigmoid
     */
    φ = this.activation_function;
    /**
     * Derivada da função de ativação sigmoid
     */
    φ_derivative(input: number): number {
        return Math.exp(-input) / Math.pow(1 + Math.exp(-input), 2)
    }
    φ_derivative_alt(input: number): number {
        return this.φ(input) * (1 - this.φ(input))
    }
    /**
     * Função de ativação sigmoid
     */
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
// uni duas arrays, tipo aquilo que tem no python
const junct = <A, B>(arr1: A[], arr2:B[]):Array<[A,B, number]> => {
    return arr1.map((e, i) => [e, arr2[i], i])
}
class OutputNeuron extends Neuron {

}
function scaleBetween(unscaledNum:number, minAllowed:number, maxAllowed:number, min:number, max:number) {
    return (maxAllowed - minAllowed) * (unscaledNum - min) / (max - min) + minAllowed;
}
function sigmoidRandom(min:number, max:number) {
    return scaleBetween(randomFromInterval(min, max), -1, 1, min, max)
}
const sum = (arr:number[]) => arr.reduce((a, b)=>a+b, 0)
// add backpropagation ou outra forma de treinar
function nn() {

    const num_iteracoes = 100000;
    const ŋ = 0.01;
    let E_log:number[] = [];

    // estrutura
    const input_layer:InputNeuron[] = [];
    const bias = new InputNeuron(0)
    input_layer.push(bias)
    bias.in_value = 1;
    const x_input = createInputNeuron()
    const y_input = createInputNeuron()
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

    let iteracao = 0;
    while (iteracao < num_iteracoes) {
        let input = [sigmoidRandom(-10, 10), sigmoidRandom(0, 80)]
        let desired = [(input[1] > (input[0]**2)) ? 1 : 0] 
        x_input.in_value = input[0];
        y_input.in_value = input[1];

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

        // Vou pular o erro global médio pois estou usando iterações de 1 em 1

        /**
         * Calculo de gradiente local dos neuronios na camada de saida
         * é bem simples já que a gente pode usar o calculo de erro com base na saida desejada
         * δ_{j}(n)=-e_{j}(n)φ'_{j}(v_{j}(n))
         * δ_{j}(n) = -(desired[j] - unit.value) * unit.φ'(unit.value)
         */
        
        const δ_saida:{ origin: Neuron, value: number }[] = []
        for (const [neuron, output, j] of junct(output_layer, output_response)) {
            const δ = -(desired[j] - output.value) * neuron.φ_derivative(output.value)
            δ_saida.push({ origin: neuron, value: δ })
        }

        // armazenar todos os deltas no mesmo lugar pra atualizar todos os pesos no fim da iteração
        const Δw_global:Array<[number, string]> = []

        /**
         * Calculo de delta de pesos entre a camada oculta e a camada de saida
         * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
         */
        // output_hidden = y_i(n)
        // Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
        for (const output_hidden of hidden_response) {
            for (const { origin, value: δ } of δ_saida) {
                const Δw = -ŋ * δ * output_hidden.value
                Δw_global.push([Δw, findLink(output_hidden.origin, origin)])
            }
        }

        
        /**
         * Calculo de gradiente local dos neuronios na camada escondida
         * δ_{j}(n)=φ'_{j}(v_{j}(n)) \sum_{i=1}^{I}δ_{i}(n)w_{ji}
         */
        const δ_hidden:{ origin: Neuron, value: number }[] = []
        for (const [neuron, output, j] of junct(hidden_layer, hidden_response)) {
            /**
             * sugestão do copilot, to sem cabeça pra saber se ta certo
            const δ = sum(Δw_global.filter(([Δw, link]) => link === findLink(neuron, output.origin)).map(([Δw, link]) => Δw))
            δ_hidden.push({ origin: neuron, value: δ })
            */

            const δ = neuron.φ_derivative(output.value) * sum(δ_saida.map(({value: grad_local, origin}, j) => grad_local * findLinkWeight(output.origin, origin)))
            δ_hidden.push({ origin: neuron, value: δ })
        }


        /**
         * Calculo de delta de pesos entre a camada de entrada e a camada oculta
         * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)=ηδ_{j}(n)x_{i}(n)
         */

        for (const neuron_input of input_layer) {
            for (const { origin, value: δ } of δ_hidden) {
                const Δw = -ŋ * δ * neuron_input.in_value
                Δw_global.push([Δw, findLink(neuron_input, origin)])
            }
        }



        for (const [ Δw, link_name ] of Δw_global) {
            links_ws[link_name] = links_ws[link_name] + Δw
        }
        console.log(E, input, desired, output_response.map(unit => unit.value))
        iteracao++
        E_log.push(E)
    }
    
    console.table({
        taxa_de_aprendizado: ŋ,
        num_iteracoes: num_iteracoes,
        primeiro_erro: E_log[0],
        erro_minimo: `${Math.min.apply(Math, E_log)} na iteração ${E_log.indexOf(Math.min.apply(Math, E_log))}`,
        erro_maximo: `${Math.max.apply(Math, E_log)} na iteração ${E_log.indexOf(Math.max.apply(Math, E_log))}`,
        erro_medio: sum(E_log)/E_log.length,
        ultimo_erro: E_log[E_log.length - 1],
    })
    console.log(chunkArray(E_log, E_log.length/8).map(chunk => sum(chunk)/chunk.length))
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