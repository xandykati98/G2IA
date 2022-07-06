// Deu tudo errado


/**
 * Objeto contendo valores dos pesos das conexões entre dois neurônios (origem e destino)
 * @example 
 * links_ws['0_to_1'] = 0.5
 * // 0 e 1 são os ids dos neurônios
 */
let links_ws: {
    [link_name:string]: number
} = {}

/**
 * unifica duas arrays, retornando uma nova array contendo todos os elementos das duas
 */
const junct = <A, B>(arr1: A[], arr2:B[]):Array<[A,B, number]> => {
    return arr1.map((e, i) => [e, arr2[i], i])
}

/**
 * Escala/reduz um número entre dois valores
 */
function scaleBetween(unscaledNum:number, minAllowed:number, maxAllowed:number, min:number, max:number) {
    return (maxAllowed - minAllowed) * (unscaledNum - min) / (max - min) + minAllowed;
}
/**
 * Retorna um valor aleatório entre min e max e o escala
 */
function sigmoidRandom(min:number, max:number) {
    return scaleBetween(randomFromInterval(min, max), 0, 1, min, max)
}
/**
 * Soma todos os valores de uma array de números
 */
const sum = (arr:number[]) => arr.reduce((a, b) => a+b, 0)

/**
 * gera um número aleatório entre um intervalo (inclusivo)
 * @param min início do intervalo
 * @param max fim do intervalo
 * @returns número aleatório
 */
function randomFromInterval(min:number, max:number) { 
    return Math.random() * (max - min) + min
}

/**
 * Cria uma nova conexão entre dois neurônios e retorna o peso da conexão
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns peso da conexão entre os dois neurônios
 */
function createLink(origin: Neuron, end: Neuron) {
    links_ws[origin.id+'_to_'+end.id] = randomFromInterval(0, 1)
    return links_ws[origin.id+'_to_'+end.id]
}

/**
 * Retorna o peso da conexão entre dois neurônios
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns peso da conexão entre os dois neurônios
 */
function findLinkWeight(origin: Neuron, end: Neuron) {
    return links_ws[origin.id+'_to_'+end.id]
}
/**
 * Retorna o nome da conexão entre dois neurônios
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns nome da conexão entre os dois neurônios
 */
function findLink(origin: Neuron, end: Neuron) {
    return origin.id+'_to_'+end.id
}

/**
 * Resultado após aplicar a função de entrada sobre os inputs do neurônio e a função de ativação sobre o resultado da função de entrada
 */
type NeuronOutput = {
    /**
     * Neurônio que originou o valor
     */
    origin: Neuron;
    /**
     * Valor da saída do neurônio
     * `y_{i}(n)`
     */
    value: number;
}

// Armazena o valor do próximo ID a ser usado em um novo neurônio
let next_id = 0;

/**
 * Classe que representa um neurônio
 */
class Neuron {
    id: number;
    constructor() {
        this.id = next_id;
        next_id++;
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

/**
 * Classe que representa neurônios de entrada
 */
class InputNeuron extends Neuron {
    in_value: number
    constructor() {
        super()
        this.in_value = 0;
    }
}

/**
 * Configuração de uma camada, utilizada na criação de camadas dinamicamente
 */
type LayerConfig = {
    /**
     * True se a camada for uma camada de entrada
     */
    is_input?: boolean,
    /**
     * True se a camada for uma camada de saída
     */
    is_output?: boolean,
    /**
     * Número de neurônios na camada
     */
    neurons_number: number,
    /**
     * Neurônio de bias
     */
    bias?: boolean,
}
/**
 * Configuração de uma camada, utilizada no processamento da rede
 */
interface LayerConfigInner extends LayerConfig {
    is_hidden: boolean
}
type Layer = {
    config: LayerConfigInner, 
    neurons: Neuron[]
};

type TrainConfig = {
    /**
     * Número de vezes que o algoritmo de treinamento será executado
     */
    epochs: number,
    /**
     * Número de vezes que o algoritmo de treinamento será executado por epoch
     */
    iteracoes: number,
    /**
     * Taxa de aprendizado da rede
     */
    taxa_aprendizado: number,
    /**
     * Funcão de utilizada ao fim de cada epoch para verificar se uma parada do treinamento deve ser feita
     */ 
    stop_condition?: (epoch: number, error: number) => boolean,
    training_set: {inputs: number[], desired_outputs: number[]}[],
}

/**
 * @note A rede só suporta 1 neuronio de bias na camada de entrada
 */
class NeuralNetwork {
    layers: Layer[];
    constructor() {
        this.layers = [];
    }
    pushLayer(layer_config: LayerConfig) {
        // Verifica se a camada é oculta
        const is_hidden = !layer_config.is_input && !layer_config.is_output;
        const NeuronType = layer_config.is_input ? InputNeuron : Neuron
        // Cria a array de neurônios da camada
        const neurons:Neuron[] = [];
        // Adiciona o bias a camada de entrada caso necessário
        if (layer_config.bias) {
            const bias = new InputNeuron();
            bias.in_value = 1;
            neurons.push(bias);
        }
        for (let i = 0; i < layer_config.neurons_number; i++) {
            neurons.push(new NeuronType())
        }
        // Adiciona a cama a rede
        this.layers.push({
            config: {...layer_config, is_hidden},
            neurons: neurons
        })
    }
    /**
     * Cria os pesos de cada conexão entre os neurônios
     */
    createWeights() {
        for (const [index, layer] of this.layers.entries()) {
            for (const neuron of layer.neurons) {
                if (!this.layers[index+1]) return;
                for (const neuron_of_next_layer of this.layers[index+1].neurons) {
                    createLink(neuron, neuron_of_next_layer)
                }
            }
        }
    }
    train_iteration(inputs:number[], desired_outputs:number[], config: TrainConfig):number {
        const input_layer = this.layers[0];
        /**
         * Adiciona os valores de entrada aos neurônios de entrada
         */
        for (const [index, neuron] of (input_layer.neurons as InputNeuron[]).entries()) {
            neuron.in_value = inputs[index];
        }
        /**
         * Salva o valor de Y de todos os neuronios de todas as camadas em ordem sequencial 
         * (a camada de saida não é considerada) 
         * (a camada de entrada é o primeiro index da array)
         */
        const layer_neuron_outputs:NeuronOutput[][] = [
            (input_layer.neurons as InputNeuron[]).map(neuron => ({ origin: neuron, value: neuron.in_value }) )
        ];

        /**
         * Alimenta os valores para frente, sempre se baseando na última camada que foi alimentada
         */
        for (const { config, neurons } of this.layers) {
            if (config.is_output) {
                // o resultado da camada de saida é criado fora do loop para preservar o valor de Y nos calculos iniciais da retropropagação
                // ps.: talvez esse valor poderia ser armazenado na variavel layer_neuron_outputs também mas escolhi assim para acessar a variavel mais facilmente
                break;
            }
            if (config.is_hidden) {
                const hidden_response:NeuronOutput[] = []
                for (const neuron of neurons) {
                    // Alimentando o resultado da ultima camada
                    hidden_response.push(neuron.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
                }
                layer_neuron_outputs.push(hidden_response);
            }
        }
        
        // Resultado da camada de saida
        const output_response:NeuronOutput[] = [];
        for (const unit of this.layers[this.layers.length - 1].neurons) {
            // Alimentando o resultado da ultima camada escondida antes da camada de saida
            output_response.push(unit.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
        }
        
        /**
         * E = Erro global instantâneo (nessa iteração)
         * E(n) = 1/2  \sum_{j=1}^{J} e^2_{j}(n)
         */
        let E = (1/2)*sum(output_response.map((unit, j) => (desired_outputs[j] - unit.value)**2))
        
        // Retropropagação

        // armazenar todos os deltas no mesmo lugar pra atualizar todos os pesos no fim da iteração
        const Δw_global:Array<[number, string]> = []

        // armazenar todos os valores das gradientes de cada camada (de forma contrária ao fluxo de dados, a camada de saida vai estar no index 0)
        const δ_layers:{ origin: Neuron, value: number }[][] = []
        
        // Ref.: https://stackoverflow.com/questions/30610523/reverse-array-in-javascript-without-mutating-original-array
        for (const [layer_index, layer] of this.layers.slice().reverse().entries()) {
            const layer_neuron_output = layer_neuron_outputs[layer_neuron_outputs.length - layer_index];
            if (layer.config.is_output) {
                /**
                 * Calculo de gradiente local dos neuronios na camada de saida
                 * é bem simples já que a gente pode usar o calculo de erro com base na saida desejada
                 * δ_{j}(n)=-e_{j}(n)φ'_{j}(v_{j}(n))
                 * δ_{j}(n) = -(desired[j] - unit.value) * unit.φ'(unit.value)
                 */
                const δ_saida:{ origin: Neuron, value: number }[] = []
                for (const [neuron, output, j]  of junct(layer.neurons, output_response)) {
                    const δ = -(desired_outputs[j] - output.value) * neuron.φ_derivative(output.value)
                    δ_saida.push({ origin: neuron, value: δ })
                }
                δ_layers.push(δ_saida)
                /**
                 * Calculo de delta de pesos entre a camada oculta antes da camada de saida e a camada de saida
                 * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
                 */
                for (const output of layer_neuron_outputs[layer_neuron_outputs.length - 1]) {
                    for (const δ of δ_saida) {
                        const Δw = -config.taxa_aprendizado * δ.value * output.value;
                        Δw_global.push([Δw, findLink(output.origin, δ.origin)])
                    }
                }
            } else if (layer.config.is_hidden) {
                /**
                 * Calculo de gradiente local dos neuronios na camada escondida (J)
                 * δ_{j}(n)=φ'_{j}(v_{j}(n)) \sum_{i=1}^{I}δ_{i}(n)w_{ji}
                 */
                const δ_escondida:{ origin: Neuron, value: number }[] = []
                //console.log(layer.config.id, ff_layers[layer_index - 1].config.id)
                for (const [neuron, output, j] of junct(layer.neurons, layer_neuron_output)) {
                    const δ = neuron.φ_derivative(output.value) * sum(δ_layers[layer_index - 1].map(({value: grad_local, origin}, j) => grad_local * findLinkWeight(output.origin, origin)))
                    δ_escondida.push({ origin: neuron, value: δ })
                }
                δ_layers.push(δ_escondida)
                
                /**
                 * Calculo de delta de pesos entre a camada (J-1) e a camada oculta (J)
                 * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)=ηδ_{j}(n)x_{i}(n)
                 */
                for (const output of layer_neuron_outputs[layer_neuron_outputs.length - 1 - layer_index]) {
                    for (const δ of δ_escondida) {
                        const Δw = -config.taxa_aprendizado * δ.value * output.value;
                        Δw_global.push([Δw, findLink(output.origin, δ.origin)])
                    }
                }
            } else if (layer.config.is_input) {
                // não faz nada
            }
        }

        for (const [ Δw, link_name ] of Δw_global) {
            links_ws[link_name]+= Δw
        }

        return E;
    }
    /**
     * Imita a propagação de uma entrada da rede até a camada de saida, retornando o resultado da saida
     * @param _inputs array de arrays de valores de entrada
     * @returns O chute da rede neural
     */
    guess(_inputs:number[]) {
        let inputs:number[] = [..._inputs];
        const input_layer = this.layers[0];
        if (input_layer.config.bias) {
            inputs.unshift(1);
        }
        for (const [i, neuron] of (input_layer.neurons as InputNeuron[]).entries()) {
            neuron.in_value = inputs[i];
        }
        const layer_neuron_outputs:NeuronOutput[][] = [(input_layer.neurons as InputNeuron[]).map(neuron => ({ origin: neuron, value: neuron.in_value }) )];
        
        for (const { config, neurons } of this.layers) {
            if (config.is_output) {
                break;
            }
            if (config.is_hidden) {
                const hidden_response:NeuronOutput[] = []
                for (const neuron of neurons) {
                    // Alimentando o resultado da ultima camada
                    hidden_response.push(neuron.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
                }
                layer_neuron_outputs.push(hidden_response);
            }
        }
        
        // Resultado da camada de saida
        const output_response:NeuronOutput[] = [];
        for (const unit of this.layers[this.layers.length - 1].neurons) {
            // Alimentando o resultado da ultima camada escondida antes da camada de saida
            output_response.push(unit.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
        }
        return output_response.map(e=>e.value);
    }

    train(config: TrainConfig) {
        const Ēs:number[] = []
        // Executa o treinamento
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            let error = 0;
            for (let i = 0; i < config.iteracoes; i++) {
                // Pega um conjunto de treinamento aleatório
                const training_item = config.training_set[Math.floor(Math.random() * config.training_set.length)]
                // Executa o treinamento
                let inputs = training_item.inputs;
                if (this.layers[0].config.bias) {
                    inputs = [1, ...inputs]
                }
                error += this.train_iteration(inputs, training_item.desired_outputs, config)
            }
            const Ē = (1/config.iteracoes)*error
            console.log({epoch, erro_medio: Ē})
            Ēs.push(Ē)
            // Verifica se a parada deve ser feita
            if (config.stop_condition && config.stop_condition(epoch, error)) break;
        }
        console.log({
            epochs: config.epochs,
            mean_error: sum(Ēs)/Ēs.length,
            std_error: Math.sqrt(sum((Ēs.map(Ē => (Ē - sum(Ēs)/Ēs.length)**2)))/Ēs.length),
            min_error: Math.min(...Ēs),
            max_error: Math.max(...Ēs),
            error_diff: Math.max(...Ēs) - Math.min(...Ēs),
            last_error: Ēs[Ēs.length - 1],
            taxa_aprendizado: config.taxa_aprendizado,
            iteracoes_por_epoch: config.iteracoes,
        })
    }
}

let rede = new NeuralNetwork()
rede.pushLayer({
    is_input: true,
    bias: true,
    neurons_number: 2,
})

rede.pushLayer({
    neurons_number:4,
})

rede.pushLayer({
    is_output: true,
    neurons_number: 1,
})

rede.createWeights()

// Criação do conjunto de treinamento
let t_set:any[] = []
while (t_set.length < 400) {
    const input = [sigmoidRandom(-10, 10), sigmoidRandom(0, 80)];
    t_set.push({
        inputs: input,
        desired_outputs: [(input[1] > (input[0]**2)) ? 1 : 0]
    })
}

rede.train({
    epochs: 3,
    iteracoes: 100000,
    taxa_aprendizado: 0.01,
    training_set: t_set,
})

// Validação
let v_set:any = []

let i_validacao = 0;

while (i_validacao < 100) {
    const input = [sigmoidRandom(-10, 10), sigmoidRandom(0, 80)];
    v_set.push({
        inputs: input,
        desired_outputs: [(input[1] > (input[0]**2)) ? 1 : 0]
    })
    i_validacao++;
}

let falso_positivos = 0;
let falso_negativos = 0;
let verdadeiros_positivos = 0;
let verdadeiros_negativos = 0;
for (const item of v_set) {
    const guess:number = rede.guess(item.inputs)[0] > 0.5 ? 1 : 0
    const expected:number = item.desired_outputs[0]
    if (guess === expected) {
        if (guess === 1) {
            verdadeiros_positivos++;
        } else {
            verdadeiros_negativos++;
        }
    } else {
        if (guess === 1) {
            falso_positivos++;
        } else {
            falso_negativos++;
        }
    }
}
const accuracy = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falso_positivos + falso_negativos);
console.log({
    verdadeiros_positivos,
    verdadeiros_negativos,
    falso_positivos,
    falso_negativos,
    accuracy
})