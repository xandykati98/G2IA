import { SemiBrief } from "./semibrief";

/**
 * Acha pares de forma aleatória, porém com um limite de distancia
 */
export class SemiBriefClose extends SemiBrief {
    reset(): void {
        this.pairs = []
        const max_distance_w = this.original_w / 4
        const max_distance_h = this.original_h / 4
        for (let i = 0; i < this.pairs_qtd; i++) {
            // Procura um pixel aleatório 
            const random_x = this.randomIntFromInterval(0, this.original_w)
            const random_y = this.randomIntFromInterval(0, this.original_h)
            // procura um pixel aleatório próximo
            const random_close_x = this.randomIntFromInterval(Math.max(0, random_x - max_distance_w), Math.min(this.original_w, random_x + max_distance_h))
            const random_close_y = this.randomIntFromInterval(Math.max(0, random_y - max_distance_h), Math.min(this.original_h, random_y + max_distance_h))

            const origin_pair = [random_x, random_y]
            const close_pair = [random_close_x, random_close_y]

            // Converter cordenadas x,y para indice
            // explicação https://catkin-clarinet-28a.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fdaefe7e5-15e5-44b4-bdcb-ad3bff1b7718%2Fconversaodecordenada.png?table=block&id=a21aef12-9fa8-4fed-a995-ec53d4fd8fa1&spaceId=88b01887-f0b9-4042-833c-9e26b475e2bf&width=2000&userId=&cache=v2
            const origin_pair_to_flattened = (this.original_w * origin_pair[1]) + origin_pair[0]
            const close_pair_to_flattened = (this.original_w * close_pair[1]) + close_pair[0]

            this.pairs.push([origin_pair_to_flattened, close_pair_to_flattened])
        }
    }
}