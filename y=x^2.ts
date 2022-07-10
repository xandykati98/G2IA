import { NeuralNetwork, sigmoidRandom } from "./nn_organized"

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
        desired_outputs: [((input[0]**2) > (input[1])) ? 1 : 0]
    })
}

rede.train({
    epochs: 60,
    iteracoes: 10000,
    taxa_aprendizado: 0.01,
    training_set: t_set,
})

// Validação
let v_set:any = [{
    inputs: [sigmoidRandom(3.6, 3.7), sigmoidRandom(17, 17.1)],
    desired_outputs: [1]
}]

let i_validacao = 0;

while (i_validacao < 1000) {
    const input = [sigmoidRandom(-10, 10), sigmoidRandom(0, 80)];
    v_set.push({
        inputs: input,
        desired_outputs: [((input[0]**2) > (input[1])) ? 1 : 0]
    })
    i_validacao++;
}

let falso_positivos = 0;
let falso_negativos = 0;
let verdadeiros_positivos = 0;
let verdadeiros_negativos = 0;
let ii = 0
for (const item of v_set) {
    const guess:number = rede.guess(item.inputs)[0] > 0.5 ? 1 : 0
    const expected:number = item.desired_outputs[0]
    if (ii === 0) {
        console.log(guess, expected)
    }
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
    ii++
}
const accuracy = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falso_positivos + falso_negativos);
console.table({
    verdadeiros_positivos,
    verdadeiros_negativos,
    falso_positivos,
    falso_negativos,
    accuracy
})
