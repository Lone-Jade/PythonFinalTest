int ai_postprocess(stai_ptr *outputs, Detection *dets, int max_dets,
                   float threshold, float nms_threshold, float min_box_size)
{
    int det_count = 0;
    int k, i, j, loc;
    int gs;
    float stride;
    float *cls, *obj, *bbox;

    for (k = 0; k < NUM_STRIDES; k++) {
        stride = (float)yu_strides[k];
        gs = yu_grid_sizes[k];
        cls = (float *)outputs[k];
        obj = (float *)outputs[k + 3];
        bbox = (float *)outputs[k + 6];

        for (i = 0; i < gs; i++) {
            for (j = 0; j < gs; j++) {
                loc = i * gs + j;
                float score = cls[loc] * obj[loc];
                if (score < threshold) continue;

                // bbox in interleaved (N×4) format, not CHW
                float dx = bbox[loc * 4 + 0];
                float dy = bbox[loc * 4 + 1];
                float dw = bbox[loc * 4 + 2];
                float dh = bbox[loc * 4 + 3];

                float cx = ((float)j + dx) * stride;
                float cy = ((float)i + dy) * stride;
                float bw = expf(dw) * stride;
                float bh = expf(dh) * stride;

                if (bw < min_box_size || bh < min_box_size) continue;

                float x1 = cx - bw * 0.5f;
                float y1 = cy - bh * 0.5f;
                float x2 = cx + bw * 0.5f;
                float y2 = cy + bh * 0.5f;

                // clip to image boundary (320×320)
                if (x1 < 0.0f) x1 = 0.0f;
                if (y1 < 0.0f) y1 = 0.0f;
                if (x2 > 320.0f) x2 = 320.0f;
                if (y2 > 320.0f) y2 = 320.0f;
                if (x2 <= x1 || y2 <= y1) continue;

                if (det_count < max_dets) {
                    dets[det_count].x1 = x1;
                    dets[det_count].y1 = y1;
                    dets[det_count].x2 = x2;
                    dets[det_count].y2 = y2;
                    dets[det_count].score = score;
                    det_count++;
                }
            }
        }
    }

    // Sort by descending score (bubble sort for brevity)
    for (int a = 0; a < det_count - 1; a++)
        for (int b = a + 1; b < det_count; b++)
            if (dets[b].score > dets[a].score) {
                Detection t = dets[a];
                dets[a] = dets[b];
                dets[b] = t;
            }

    // NMS
    for (int a = 0; a < det_count; a++) {
        if (dets[a].score <= 0.0f) continue;
        float area_a = (dets[a].x2 - dets[a].x1) * (dets[a].y2 - dets[a].y1);
        for (int b = a + 1; b < det_count; b++) {
            if (dets[b].score <= 0.0f) continue;
            float xx1 = fmaxf(dets[a].x1, dets[b].x1);
            float yy1 = fmaxf(dets[a].y1, dets[b].y1);
            float xx2 = fminf(dets[a].x2, dets[b].x2);
            float yy2 = fminf(dets[a].y2, dets[b].y2);
            float iw = xx2 - xx1, ih = yy2 - yy1;
            if (iw <= 0.0f || ih <= 0.0f) continue;
            float inter = iw * ih;
            float area_b = (dets[b].x2 - dets[b].x1) * (dets[b].y2 - dets[b].y1);
            float iou = inter / (area_a + area_b - inter);
            float smaller_area = (area_a < area_b) ? area_a : area_b;
            if (iou > nms_threshold || inter > 0.75f * smaller_area)
                dets[b].score = -1.0f;
        }
    }

    // Compact list
    int keep = 0;
    for (int a = 0; a < det_count; a++)
        if (dets[a].score > 0.0f)
            dets[keep++] = dets[a];

    return keep;
}