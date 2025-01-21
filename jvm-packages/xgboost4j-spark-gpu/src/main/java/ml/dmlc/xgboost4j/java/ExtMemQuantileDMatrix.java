package ml.dmlc.xgboost4j.java;

import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.core.JsonProcessingException;

import java.util.Iterator;

public class ExtMemQuantileDMatrix extends DMatrix {
  public ExtMemQuantileDMatrix(Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      DMatrix ref,
      int nthread,
      int max_num_device_pages,
      int max_quantile_batches,
      int min_cache_page_bytes) {
    super(0);
    long[] out = new long[1];
    long[] ref_handle = null;
    if (ref != null) {
      ref_handle = new long[1];
      ref_handle[0] = ref.getHandle();
    }
    String conf = getConfig(missing, maxBin, nthread);
    XGBoostJNI.checkCall(XGBoostJNI.XGQuantileDMatrixCreateFromCallback(
        iter, ref, conf, out));
    handle = out[0];
  }

  private String getConfig(float missing, int maxBin, int nthread) {
    Map<String, Object> conf = new java.util.HashMap<>();
    conf.put("missing", missing);
    conf.put("max_bin", maxBin);
    conf.put("nthread", nthread);
    conf.put("use_ext_mem", true);
    ObjectMapper mapper = new ObjectMapper();

    // Handle NaN values. Jackson by default serializes NaN values into strings.
    SimpleModule module = new SimpleModule();
    module.addSerializer(Double.class, new F64NaNSerializer());
    module.addSerializer(Float.class, new F32NaNSerializer());
    mapper.registerModule(module);

    try {
      String config = mapper.writeValueAsString(conf);
      return config;
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize configuration", e);
    }
  }
};
