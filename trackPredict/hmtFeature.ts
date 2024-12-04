import configs from "@config";
import * as tf from "@tensorflow/tfjs";
import { ulid } from "ulid";
import munkres from "munkres";
import { storeId } from "../util/util.js";
import { HmtFeature, HmtExtract, Extract } from "../types.js";
import { iouTensor } from './trackPredictUtil.js';
import { KalmanFilter } from '../trackPredict/kalmanFilter_tfjs.js';

function getHmtFeature(
  id: string,
  feature: number[],
  bbox : number [],
  kf : KalmanFilter,
  skipCount : number,
  createdAt = new Date(),
  updatedAt = new Date()
): HmtFeature {
  return { id, storeId, bbox, kf, feature, skipCount, createdAt, updatedAt };
}

function kalmanFilterInitialize(extractMap : Map<number, Extract>){
  tf.engine().startScope();
  const kfBboxMap: Map<number, KalmanFilter> = new Map();
  for (const [idx, extractObj] of extractMap) {
    const kf = new KalmanFilter(extractObj.bbox);
    kf.predict();
    kfBboxMap.set(idx, kf);
  }
  tf.engine().endScope();

  return kfBboxMap;
}

export function updateHmtFeature(
  hmtFeatureMap: Map<string, HmtFeature>,
  extractMap: Map<number, Extract>
) {
  const hmtExtractMap: Map<number, HmtExtract> = new Map();

  // hmtFeatureMap이 없을 경우 => initialize
  if (hmtFeatureMap.size == 0) {
    const kfBboxMap = kalmanFilterInitialize(extractMap);
    for (const [idx, extractObj] of extractMap) {
      const id = ulid();
      const hmtFeature = getHmtFeature(id, extractObj.feature, extractObj.bbox, kfBboxMap.get(idx)!, 0);
      hmtFeatureMap.set(id, hmtFeature);

      // hmtExtractObj:image crop용 
      const hmtExtractObj: HmtExtract = {
        feature: hmtFeature,
        bbox: extractObj.bbox,
      };
      if (extractObj.cropImg) hmtExtractObj.cropImg = extractObj.cropImg;
      hmtExtractMap.set(idx, hmtExtractObj);
    }
    return hmtExtractMap;
  }

  else if (extractMap.size === 0 && hmtFeatureMap.size !== 0){
    for (const [idx, extractObj] of hmtFeatureMap){
      if (extractObj.skipCount > 10){
        const skipCount = 0;
        const id = extractObj.id as string;
        const hmtFeature = getHmtFeature(id, extractObj.feature, extractObj.bbox, extractObj.kf, skipCount);
        hmtFeatureMap.set(id, hmtFeature);
        continue;
      } 
      extractObj.kf.update(extractObj.bbox);
      extractObj.kf.predict();
      const id = extractObj.id as string;
      const skipCount = extractObj.skipCount + 1;
      const newBbox = extractObj.kf.prediction;
      const hmtFeature = getHmtFeature(id, extractObj.feature, newBbox, extractObj.kf, skipCount);
      hmtFeatureMap.set(id, hmtFeature);
    }
  }

  // 1단계 => 칼만 필터를 통해 모든 bbox의 iou를 측정
  // 2단계 => 매칭되지 않은 extractMap은 feature matching하여 유사한 것을 찾는다 => 그래도 매칭되지 않는 것은 새롭게 아이디 부여
  // 3단계 => 매칭되지 않은 hmtfeatureMap은 삭제 혹은 보류, kf.update(previous this.prediction) 수행하여 공분산/오차 공분산 업데이트

  tf.engine().startScope();
  // feature 유사도 측정
  const featureCosts = cosineSimilarity(hmtFeatureMap, extractMap);

  const hmtBbox: number[][] = [];
  for (const f of hmtFeatureMap.values()) {
    if (f.kf && f.kf.prediction) {
      hmtBbox.push(f.kf.prediction);
    }
  }
  const extractBbox = Array.from(extractMap.values()).map((e) => e.bbox);
  const iouCosts : number[][] = iouTensor(hmtBbox, extractBbox);

  // 객체 탐지 값이 1이고 cost값이 임계치 이상일 경우 => 아이디 새로 부여
  if (iouCosts.length === 1 && iouCosts[0][0] > 0.2) {
    const id = ulid();
    const kf = new KalmanFilter(extractMap.get(0)!.bbox);
    const hmtFeature = getHmtFeature(id, extractMap.get(0)!.feature, extractMap.get(0)!.bbox, kf, 0);
    const oldFeatureId = Array.from(hmtFeatureMap.keys())[0];
    hmtFeatureMap.delete(oldFeatureId);
    hmtFeatureMap.set(id, hmtFeature);

    hmtExtractMap.set(0, {
      feature: hmtFeature,
      cropImg: extractMap.get(0)!.cropImg,
      bbox: extractMap.get(0)!.bbox,
    });
    return hmtExtractMap;
  }

  // const hungarianResults = munkres(featureCosts);
  const hungarianResults = munkres(iouCosts);
  const hmtFeatureArray = Array.from(hmtFeatureMap.values());
  const hmtFeatureKeys = Array.from(hmtFeatureMap.keys());

  const matchedRow : number[] = [];
  const matchedCol : number[] = [];
  hungarianResults.forEach(([row, col]) => {
    const extractObj = extractMap.get(col);
    matchedRow.push(row);
    matchedCol.push(col);
    

    // 매칭이 이루어지는 경우 => bbox, feature, kf 업데이트
    // if (featureCosts[row][col] < configs.threshold) {
    if (iouCosts[row][col] < 0.3) {
      const currentFeature = hmtFeatureArray[row];
      currentFeature.kf.update(extractObj!.bbox);
      currentFeature.feature = extractObj!.feature;
      currentFeature.updatedAt = new Date();
      hmtFeatureMap.set(hmtFeatureKeys[row], currentFeature);

      hmtExtractMap.set(col, {
        feature: currentFeature,
        cropImg: extractObj!.cropImg,
        bbox: extractObj!.bbox,
      });
    } 

    //------------------------------------------------------------------
    // 매칭이 이루어지지 않는 경우
    else {
      const hmtFeatureId = ulid();
      const kf = new KalmanFilter(extractObj!.bbox);
      const hmtFeature = getHmtFeature(hmtFeatureId, extractObj!.feature, extractObj!.bbox, kf, 0);
      hmtFeatureMap.set(hmtFeatureId, hmtFeature);
      hmtExtractMap.set(col, {
        feature: hmtFeature,
        cropImg: extractObj!.cropImg,
        bbox: extractObj!.bbox,
      });


    }
  });
  tf.engine().endScope();

  return hmtExtractMap;
}

export function featureAndCropImgExtract(
  img: tf.Tensor<tf.Rank.R4>,
  faceBboxes: number[][],
  extractor: tf.LayersModel
) {
  const imgInputShape = configs.imgInputShape;
  tf.engine().startScope();
  const extractMap = new Map();
  if (faceBboxes.length !== 0) {
    let idx = 0;
    for (const fb of faceBboxes) {
      const xVal = fb[0];
      const yVal = fb[1];
      const wVal = fb[2];
      const hVal = fb[3];

      const x = Math.min(Math.max(xVal, 0), imgInputShape[0]);
      const y = Math.min(Math.max(yVal, 0), imgInputShape[1]);
      const w = Math.min(wVal, imgInputShape[0] - x);
      const h = Math.min(hVal, imgInputShape[1] - y);

      tf.tidy(() => {
        const sliced = tf.slice(img, [0, y, x, 0], [1, h, w, 3]);
        const cap2 = tf.image.resizeBilinear(sliced, [256, 128]);

        const normalizedSlice = tf.mul(cap2, 255);
        const uint8Slice = tf.cast(normalizedSlice, "int32");
        const slicedArr = new Uint8Array(tf.squeeze(uint8Slice).dataSync());

        const predictRes = extractor.predict(cap2) as tf.Tensor<tf.Rank.R1>;
        const featureArr = predictRes.flatten().arraySync();

        if (featureArr && fb && slicedArr) {
          extractMap.set(idx, {
            feature: featureArr,
            cropImg: slicedArr,
            bbox: fb,
          });
        }
      });
      idx++;
    }
  }
  tf.engine().endScope();
  return extractMap;  // Map<[id, {feature, cropImg, bbox}], [id, {feature, cropImg, bbox}], [id, {feature, cropImg, bbox}]...>
}


//삭제

function cosineSimilarity(hmtFeatureMap: Map<string, HmtFeature>, extractMap: Map<number ,Extract>) : number[][] {
  const hmtFeatureArray = Array.from(hmtFeatureMap.values()).map(
    (f) => f.feature
  );
  const extractFeatureArray = Array.from(extractMap.values()).map(
    (e) => e.feature
  );
  const featureCosts: number[][] = tf.tidy(() => {
    //feature 길이 : [1,512]
    const featureLength = hmtFeatureArray[0].length;
    const hmtTensor : tf.Tensor2D = tf.tensor2d(hmtFeatureArray, [
      hmtFeatureArray.length,
      featureLength,
    ]);
    const extractTensor : tf.Tensor2D = tf.tensor2d(extractFeatureArray, [
      extractFeatureArray.length,
      featureLength,
    ]);

    const dot = tf.matMul(hmtTensor, extractTensor.transpose());
    const hmtNorm = tf.norm(hmtTensor, 2, 1, true);
    const extractNorm = tf.norm(extractTensor, 2, 1, true);
    const normMatrix = tf.matMul(hmtNorm, extractNorm.transpose());

    const similarities : tf.Tensor2D = tf.div(dot, normMatrix);
    const output : tf.Tensor2D = tf.sub(tf.ones(similarities.shape), similarities);
    const outputArr : number[][] = output.arraySync();
    return outputArr 
  });

  return featureCosts;
}

// // 피어슨 상관 계수를 계산하는 함수
// function pearsonCorrelation(vecA: number[], vecB: number[]) {
//   const correlation = tf.tidy(() => {
//     const meanA = tf.mean(vecA);
//     const meanB = tf.mean(vecB);

//     const diffA = tf.sub(vecA, meanA);
//     const diffB = tf.sub(vecB, meanB);

//     const numerator = tf.sum(tf.mul(diffA, diffB));

//     const denominator = tf.mul(
//       tf.sqrt(tf.sum(tf.square(diffA))),
//       tf.sqrt(tf.sum(tf.square(diffB)))
//     );

//     const correlation = tf.div(numerator, denominator);
//     const correlationArr = correlation.dataSync()[0];
//     correlation.dispose();
//     meanA.dispose();
//     meanB.dispose();
//     diffA.dispose();
//     diffB.dispose();
//     return correlationArr;
//   });

//   const rescale = 1 - Math.abs(correlation);

//   return rescale;
// }
