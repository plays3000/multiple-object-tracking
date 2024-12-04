import * as tf from '@tensorflow/tfjs'
import {getMatrixInverse} from '../util/linalg.js';

export class KalmanFilter {
  dt: number;
  stateVariance: number;
  measurementVariance: number;
  method: string;
  prediction! : number[];
  U! : number;
  errorCov! : number[][];
  state! : number[][];
  predictedState! : number[][];
  predictedErrorCov! : number[][];

  constructor(
    detection: number[],
    dt: number = 1,
    stateVariance: number = 1,
    measurementVariance: number = 1,
    method: string = "Velocity",
  ) {
    this.method = method;
    this.stateVariance = stateVariance;
    this.measurementVariance = measurementVariance;
    this.dt = dt;
    this.initModel(detection);
  }

  initModel(detection: number[]): void {
    if (this.method === "Acceleration") {
      this.U = 1;
    } else {
      this.U = 0;
    }

    // 오차 공분산 행렬 초기화 // 8x8
    this.errorCov = [
      [this.stateVariance,0,0,0,0,0,0,0],
      [0,this.stateVariance,0,0,0,0,0,0],
      [0,0,this.stateVariance,0,0,0,0,0],
      [0,0,0,this.stateVariance,0,0,0,0],
      [0,0,0,0,this.stateVariance,0,0,0],
      [0,0,0,0,0,this.stateVariance,0,0],
      [0,0,0,0,0,0,this.stateVariance,0],
      [0,0,0,0,0,0,0,this.stateVariance]
    ];

    // 상태 벡터 초기화 [x, vx, y, vy, w, vw, h, vh]
    this.state = [
          [detection[0]],  // x 위치
          [0.0],           // x 속도
          [detection[1]],  // y 위치
          [0.0],           // y 속도
          [detection[2]],  // w (폭)
          [0.0],           // w 변화율
          [detection[3]],  // h (높이)
          [0.0]            // h 변화율
      ]
  }

  predict(): void{
    const dt = this.dt;
    const A = tf.tensor2d(
      [
        [1, dt, 0,  0,  0,  0,  0,  0],
        [0, 1,  0,  0,  0,  0,  0,  0],
        [0, 0,  1, dt, 0,  0,  0,  0],
        [0, 0,  0, 1,  0,  0,  0,  0],
        [0, 0,  0, 0,  1, dt, 0,  0],
        [0, 0,  0, 0,  0, 1,  0,  0],
        [0, 0,  0, 0,  0, 0,  1, dt],
        [0, 0,  0, 0,  0, 0,  0, 1]
      ],
      [8, 8],
      'float32'
    );

    const B = tf.tensor2d(
      [
        [(dt ** 2) / 2],
        [dt],
        [(dt ** 2) / 2],
        [dt],
        [(dt ** 2) / 2],
        [dt],
        [(dt ** 2) / 2],
        [dt],
      ],
      [8, 1],
      'float32'
    );

    const Q : tf.Tensor2D = tf.mul(
      this.stateVariance,
      tf.tensor2d(
        [
            [(dt ** 4) / 4, (dt ** 3) / 2, 0,             0,             0,             0,             0,             0],
            [(dt ** 3) / 2, dt ** 2,       0,             0,             0,             0,             0,             0],
            [0,             0,             (dt ** 4) / 4, (dt ** 3) / 2, 0,             0,             0,             0],
            [0,             0,             (dt ** 3) / 2, dt ** 2,       0,             0,             0,             0],
            [0,             0,             0,             0,             (dt ** 4) / 4, (dt ** 3) / 2, 0,             0],
            [0,             0,             0,             0,             (dt ** 3) / 2, dt ** 2,       0,             0],
            [0,             0,             0,             0,             0,             0,             (dt ** 4) / 4, (dt ** 3) / 2],
            [0,             0,             0,             0,             0,             0,             (dt ** 3) / 2, dt ** 2],
        ],
        [8,8],
        'float32'
      )
    );

    // predictedState = A * state + B * U
    const matMul : tf.Tensor2D = tf.matMul(A, this.state);
    const mul : tf.Tensor2D = tf.mul(B, this.U);
    const predictedState : tf.Tensor2D = tf.add(matMul, mul);
    const predictedStateArr : number[][] = predictedState.arraySync();
    this.predictedState = predictedStateArr;

    // predictedErrorCov = A * errorCov * A' + Q  => 8x8
    const matMul1 : tf.Tensor2D = tf.matMul(A, this.errorCov);
    const matMul2 : tf.Tensor2D = tf.matMul(matMul1, A.transpose());
    const add : tf.Tensor2D =  tf.add(matMul2, Q);
    const addArr : number[][] = add.arraySync();
    this.predictedErrorCov = addArr;

    // this.predictedErrorCov = tf.add(tf.matMul(tf.matMul(A, this.errorCov), tf.transpose(A)),Q);

    // 예측된 위치 추출
    const x = Number(this.predictedState[0][0]);
    const y = Number(this.predictedState[2][0]);
    const w = Number(this.predictedState[4][0]);
    const h = Number(this.predictedState[6][0]);

    // 메모리 해제
    A.dispose();
    B.dispose();
    Q.dispose();

    matMul.dispose();
    mul.dispose();
    predictedState.dispose();
    matMul1.dispose();
    matMul2.dispose();
    add.dispose();

    this.prediction = [x, y, w, h];
  }

  update(currentMeasurement: number[]): void {
    
    const H = tf.tensor2d(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],  // x 위치
        [0, 0, 1, 0, 0, 0, 0, 0],  // y 위치
        [0, 0, 0, 0, 1, 0, 0, 0],  // w (폭)
        [0, 0, 0, 0, 0, 0, 1, 0],  // h (높이)
    ],
    [4, 8],
    'float32'
    );

    const z = tf.tensor2d([
        [currentMeasurement[0]],  // x 위치
        [currentMeasurement[1]],  // y 위치
        [currentMeasurement[2]],  // w (너비)
        [currentMeasurement[3]]   // h (높이)
        ],
        [4,1],
        'float32'
    )

    const K = tf.tidy(()=>{
        const R = tf.mul(this.measurementVariance, tf.eye(4));
        const S : tf.Tensor2D= tf.add(tf.matMul(tf.matMul(H, this.predictedErrorCov), H.transpose()), R);
        const sInv = getMatrixInverse(S);
        const K : tf.Tensor2D = tf.matMul(tf.matMul(this.predictedErrorCov, tf.transpose(H)),sInv);

        return K;
    })
    
    // y = z - (H @ this.predicted_state)
    const matMul1 = tf.matMul(H, this.predictedState);
    const sub1 = tf.sub(z, matMul1);
    const y = tf.sub(z, tf.matMul(H, this.predictedState));
    this.state = tf.add(this.predictedState, tf.matMul(K, y)).arraySync() as number[][];

    const I = tf.eye(this.state.length);

    // this.errorCov = (I - (K @ self.H) @ self.predicted_error_cov)
    this.errorCov = tf.matMul(tf.sub(I, tf.matMul(K, H)),this.predictedErrorCov).arraySync() as number[][];

    // 메모리 해제
    K.dispose();
    H.dispose();
    y.dispose();
    I.dispose();
    matMul1.dispose();
    sub1.dispose();
  }
    
}
