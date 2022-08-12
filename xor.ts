import { benchmark } from "./benchmark";
import { links_ws, NeuralNetwork } from "./nn_organized";

const rede = new NeuralNetwork();
rede.pushLayer({
    is_input: true,
    neurons_number: 2,
    bias: true
})
rede.pushLayer({
    neurons_number: 4,
})
rede.pushLayer({
    is_output: true,
    neurons_number: 1,
})

const train_set:any[] = [
    { inputs: [0, 0, 1], desired_outputs: [0] },
    { inputs: [0, 1, 1], desired_outputs: [1] },
    { inputs: [1, 0, 1], desired_outputs: [1] },
    { inputs: [1, 1, 1], desired_outputs: [0] },
]
//while (train_set.length < 10000) {
//    const x = Math.random() > 0.5 ? 1 : 0;
//    const y = Math.random() > 0.5 ? 1 : 0;
//
//    train_set.push({
//        inputs: [x, y],
//        desired_outputs: [x ^ y],
//    });
//}

const test_set:{inputs: number[], desired_outputs: number[]}[] = []
while (test_set.length < 200) {
    const x = Math.random() > 0.5 ? 1 : 0;
    const y = Math.random() > 0.5 ? 1 : 0;

    test_set.push({
        inputs: [x, y],
        desired_outputs: [x ^ y],
    });
}
const train_config = {
    epochs: 90,
    momentum: 0.01,
    iteracoes: 10000,
    taxa_aprendizado: 0.01,
    training_set: train_set,
    silent: true
}
//rede.train(train_config)

benchmark({
    v_set: test_set,
    train_config: train_config,
    rede: rede,
    get_prediction: output => Math.round(output),
    bechnmark_name: "xor",
})

/*
let falso_positivos = 0;
let falso_negativos = 0;
let verdadeiros_positivos = 0;
let verdadeiros_negativos = 0;
rede.test(test_set, (error, output, desired) => {
    const prediction = Math.round(output[0])
    const expected = Math.round(desired[0])
    if (prediction === expected) {
        if (prediction === 1) {
            verdadeiros_positivos++;
        } else {
            verdadeiros_negativos++;
        }
    } else {
        if (prediction === 1) {
            falso_positivos++;
        } else {
            falso_negativos++;
        }
    }
});
const accuracy = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falso_positivos + falso_negativos);
console.log({
    verdadeiros_positivos,
    verdadeiros_negativos,
    falso_positivos,
    falso_negativos,
    accuracy,
})*/