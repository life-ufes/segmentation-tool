import { fromEvent } from rxjs;
import { switchMap, takeUntil, pairwise, map } from rxts.operators;

export class CanvasComponent {

    constructor(imagemService) {
        this.imagemService = imagemService;
        this.imgDim = undefined;
        this.respUsuario = undefined;
    }

    ngAfterViewInit() {
        let idx = this.imagemPath.indexOf('.png');
        if (idx !== -1) {
            this.imagemPath = this.imagemPath.substring(0, idx);
        }

        const imagem = new Image();
        imagem.src = this.imagemUrl;
        imagem.onload = () => {
            this.imgDim = {
                largura: imagem.naturalWidth,
                altura: imagem.naturalHeight
            };
        };

        const canvas_seg = document.getElementById('img_seg');
        canvas_seg.width = window.innerWidth * 0.35;
        canvas_seg.height = canvas_seg.width;

        const canvas_res = document.getElementById('img_res');
        canvas_res.width = window.innerWidth * 0.35;
        canvas_res.height = canvas_res.width;

        this.cx_seg = canvas_seg.getContext('2d');
        this.cx_seg.lineWidth = 2;
        this.cx_seg.lineCap = 'round';
        this.cx_seg.strokeStyle = 'white';

        this.cx_res = canvas_res.getContext('2d');
        this.cx_res.fillStyle = 'white';
        this.cx_res.fillRect(0, 0, canvas_res.width, canvas_res.height);

        let mask = new Image();
        mask.crossOrigin = "anonymous";
        idx = this.imagemUrl.indexOf(this.tipoPaciente) + this.tipoPaciente.length;
        mask.onload = () => {
            this.cx_res.drawImage(mask, 0, 0, mask.naturalWidth, mask.naturalHeight,
                                        0, 0, canvas_res.width, canvas_res.height);
        };
        mask.src = this.imagemUrl.substring(0, idx) + '-mask' + this.imagemUrl.substring(idx);

        this.ferramentaMaoLivre(canvas_seg);
    }

    ferramentaMaoLivre(canvas_seg) {
        let initPos;

        fromEvent(canvas_seg, 'mousedown')
        .pipe(
            switchMap((e) => {
            initPos = undefined;
            return fromEvent(canvas_seg, 'mousemove')
                .pipe(
                    takeUntil(fromEvent(canvas_seg, 'mouseup').pipe(
                        map( () => {
                            this.cx_seg.closePath();
                            this.cx_seg.stroke();

                            this.cx_seg.fillStyle = 'white';
                            this.cx_seg.fill();

                            this.cx_res.clearRect(0, 0, canvas_seg.width, canvas_seg.height);
                            this.cx_res.fillStyle = 'black';
                            this.cx_res.fillRect(0, 0, canvas_seg.width, canvas_seg.height);

                            this.cx_res.drawImage(canvas_seg, 0, 0);

                            this.cx_seg.clearRect(0, 0, canvas_seg.width, canvas_seg.height);
                        } )
                    )),
                    pairwise()
                );
            })
        )
        .subscribe((res) => {
            const rect = canvas_seg.getBoundingClientRect();

            const posAnterior = {
            x: res[0].clientX - rect.left,
            y: res[0].clientY - rect.top
            };

            const posAtual = {
            x: res[1].clientX - rect.left,
            y: res[1].clientY - rect.top
            };

            if (initPos === undefined) {
                initPos = posAnterior;
                this.cx_seg.beginPath();
                this.cx_seg.moveTo(initPos.x, initPos.y);
            }

            this.desenharNoCanvas(posAnterior, posAtual);
        });
    }

    desenharNoCanvas(posAnterior, posAtual) {
        if (!this.cx_seg) { return; }

        if (posAnterior) {
            this.cx_seg.lineTo(posAtual.x, posAtual.y);
            this.cx_seg.stroke();
        }
    }

    cancelarSegmentacao() {
        this.eventoCancelarSegmentacao.emit();
    }

    salvarMascara() {
        this.respUsuario = 'salvar-mascara-aguarde';

        const canvas_res = document.getElementById('img_res');

        let canvas_copy = document.createElement('canvas');
        let ctx = canvas_copy.getContext('2d');

        canvas_copy.width = this.imgDim.largura;
        canvas_copy.height = this.imgDim.altura;
        ctx.drawImage(canvas_res, 0, 0, this.imgDim.largura, this.imgDim.altura);
        const imagemPng = canvas_copy.toDataURL();

        this.imagemService.uploadImagemBase64(imagemPng, this.tipoPaciente + '-mask', this.imagemPath).subscribe(
            resp => {
                this.respUsuario = 'salvar-mascara-ok';

                let url;
                if (this.tipoPaciente === 'dermato') {
                    url = URL_API + '/api/imagemDermato/' + this.imagemId;
                } else {
                    url = URL_API + '/api/imagemCirurgia/' + this.imagemId;
                }
                const dados = { 'segmentado': true };
                this.imagemService.atualizarImagem(url, dados).subscribe(
                    resp => {
                        console.log('Tabela atualizada corretamente.');
                    },
                    erro => {
                        console.log('Erro ao atualizar tabela imagem.');
                    }
                );
            },
            erro => {
                if (erro.status === 500) {
                    console.log("Error 500");
                }
                                this.respUsuario = 'salvar-mascara-erro';
            }
        );
    }
}
