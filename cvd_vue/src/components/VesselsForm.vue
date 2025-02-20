<script>
export default {
  name: 'VesselsForm',
  props: {
    waitingForResponse: {
      type: Boolean,
      defauls: false,
    },
  },
  data() {
    return {
      form: {
        SpeedMinimum: 0,
        SpeedQ1: 0,
        SpeedMedian: 0,
        SpeedQ3: 0,
        DistanceStartShapeCurvature: 52.4,
        DistanceStartTrendAngle: 0,
        DistStartTrendDevAmplitude: 0,
        MaxDistPort: 0,
        MinDistPort: 0,
      },
      neighb_parameters: {
        NeighbTypes: [],
        NumSamples: 3000,
      },
      neighb_types_options: [
        { text: 'Training set', value: 'train' },
        { text: 'Random generator', value: 'random' },
        { text: 'Custom', value: 'custom' },
        { text: 'Genetic', value: 'genetic' },
        { text: 'Custom genetic', value: 'custom_genetic' },
        { text: 'Baseline Train', value: 'baseline' },
      ],
      relevant_instances: [
        {
          title: 'Class 1 - Speeds Varied from 14.35 to 18.95 kts, Initial Curvature of 1 Unit, and Distances Closest to Port Ranging from 0.27 to 34.35 nautical miles.',
          values: {
            SpeedMinimum: 14.35,
            SpeedQ1: 15.46,
            SpeedMedian: 16.5,
            SpeedQ3: 18.95,
            DistanceStartShapeCurvature: 1,
            DistanceStartTrendAngle: 0.27,
            DistStartTrendDevAmplitude: 1.71,
            MaxDistPort: 34.35,
            MinDistPort: 22.19,
          },
        },
        {
          title: 'Class 3 - Speeds Ranged from 3.27 to 4.82 kts, Initial Path Curvature of 1 Unit, and Minimum Distances to Port were 0.08 to 0.17 nautical miles',
          values: {
            SpeedMinimum: 3.27,
            SpeedQ1: 4.58,
            SpeedMedian: 4.68,
            SpeedQ3: 4.82,
            DistanceStartShapeCurvature: 1,
            DistanceStartTrendAngle: 0.08,
            DistStartTrendDevAmplitude: 0.17,
            MaxDistPort: 63.35,
            MinDistPort: 48.87,
          },
        },
        {
          title: 'Class 3 - Record Exhibits Speeds from 0.91 to 2.78 kts, Initial Path Curvature of 2.45 Units, and Distances Closest to Port at 5.07 to 27.55 nautical miles.',
          values: {
            SpeedMinimum: 0.91,
            SpeedQ1: 2.37,
            SpeedMedian: 2.49,
            SpeedQ3: 2.78,
            DistanceStartShapeCurvature: 2.45,
            DistanceStartTrendAngle: -0.01,
            DistStartTrendDevAmplitude: 5.07,
            MaxDistPort: 27.55,
            MinDistPort: 20.04,
          },
        },
      ],
    };
  },
  methods: {
    sendRequest() {
      const request = {
        vessel_event: {
          SpeedMinimum: this.form.SpeedMinimum,
          SpeedQ1: this.form.SpeedQ1,
          SpeedMedian: this.form.SpeedMedian,
          SpeedQ3: this.form.SpeedQ3,
          DistanceStartShapeCurvature: this.form.DistanceStartShapeCurvature,
          DistanceStartTrendAngle: this.form.DistanceStartTrendAngle,
          DistStartTrendDevAmplitude: this.form.DistStartTrendDevAmplitude,
          MaxDistPort: this.form.MaxDistPort,
          MinDistPort: this.form.MinDistPort,
        },
        num_samples: this.neighb_parameters.NumSamples,
        neighborhood_types: this.neighb_types_options
          .filter(v => this.neighb_parameters.NeighbTypes.includes(v.value))
          .map(v => v.value),
      };
      console.log('form', request);
      const strValue = JSON.stringify(request);
      this.$emit('sendRequest', strValue);
    },
  },
};
</script>

<template>
  <b-form>
    <b-row>
      <b-col>
        <b-form-group id="relevant-instances-g" label-for="relevant-instances-i"
        description="Select one of the relevant instances to fill the form">
          <label for="relevant-instances-i">Relevant instances:</label>
          <b-form-select id="relevant-instances-i" v-model="form" required
          :options="relevant_instances.map(v => ({ text: v.title, value: v.values }))"/>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col cols="6">
        <b-form-group id="SpeedMinimum-g" label-for="SpeedMinimum-i"
        description="The minimum speed in the period">
          <label for="SpeedMinimum-i">Speed Minimum: {{form.SpeedMinimum}}</label>
          <b-form-input id="SpeedMinimum-i" v-model="form.SpeedMinimum" required
          type="range" :min="0" :max="20" :step="0.01"/>
        </b-form-group>
      </b-col>
      <b-col cols="6">
        <b-form-group id="SpeedQ1-g" label-for="SpeedQ1-i"
                      description="The first quartile of alue in the period">
          <label for="SpeedQ1-i">Speed Q1: {{form.SpeedQ1}}</label>
          <b-form-input id="SpeedQ1-i" v-model="form.SpeedQ1" required
          type="range" :min="0" :max="22" :step="0.01"/>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col cols="6">
        <b-form-group id="SpeedMedian-g" label-for="SpeedMedian-i"
        description="The median speed in the period">
          <label for="SpeedMedian-i">Speed Median: {{form.SpeedMedian}}</label>
          <b-form-input id="SpeedMedian-i" v-model="form.SpeedMedian" required
          type="range" :min="0" :max="22" :step="0.01"/>
        </b-form-group>
      </b-col>
      <b-col cols="6">
        <b-form-group id="SpeedQ3-g" label-for="SpeedQ3-i"
                      description="The third quartile of alue in the period">
          <label for="SpeedQ1-i">Speed Q3: {{form.SpeedQ3}}</label>
          <b-form-input id="SpeedQ3-i" v-model="form.SpeedQ3" required
          type="range" :min="0" :max="22" :step="0.01"/>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col cols="4">
        <b-form-group id="DistanceStartShapeCurvature-g" label-for="DistanceStartShapeCurvature-i"
        description="The distance start shape curvature">
          <label for="DistanceStartShapeCurvature-i">DistanceStartShapeCurvature: {{form.DistanceStartShapeCurvature}}</label>
          <b-form-input id="DistanceStartShapeCurvature-i" v-model="form.DistanceStartShapeCurvature" required
          type="range" :min="1" :max="180" :step="0.1"/>
        </b-form-group>
      </b-col>
      <b-col cols="4">
        <b-form-group id="DistanceStartTrendAngle-g" label-for="DistanceStartTrendAngle-i"
                      description="The trend angle">
          <label for="DistanceStartTrendAngle-i">DistanceStartTrendAngle: {{form.DistanceStartTrendAngle}}</label>
          <b-form-input id="DistanceStartTrendAngle-i" v-model="form.DistanceStartTrendAngle" required
          type="range" :min="-2" :max="2" :step="0.01"/>
        </b-form-group>
      </b-col>
      <b-col cols="4">
        <b-form-group id="DistStartTrendDevAmplitude-g" label-for="DistStartTrendDevAmplitude-i"
                      description="The trend angle">
          <label for="DistStartTrendDevAmplitude-i">DistStartTrendDevAmplitude: {{form.DistStartTrendDevAmplitude}}</label>
          <b-form-input id="DistStartTrendDevAmplitude-i" v-model="form.DistStartTrendDevAmplitude" required
          type="range" :min="0" :max="60" :step="0.01"/>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col cols="6">
        <b-form-group id="MaxDistPort-g" label-for="MaxDistPort-i"
        description="The maximum distance from port in the period">
          <label for="MaxDistPort-i">Maximum distance from port: {{form.MaxDistPort}}</label>
          <b-form-input id="MaxDistPort-i" v-model="form.MaxDistPort" required
          type="range" :min="0" :max="300" :step="0.1"/>
        </b-form-group>
      </b-col>
      <b-col cols="6">
        <b-form-group id="MinDistPort-g" label-for="MinDistPort-i"
                      description="The minimum distance from port in the period">
          <label for="MinDistPort-i">Minimum distance from port: {{form.MinDistPort}}</label>
          <b-form-input id="MinDistPort-i" v-model="form.MinDistPort" required
          type="range" :min="0" :max="300" :step="0.1"/>
        </b-form-group>
      </b-col>
    </b-row>
    <hr/>
    <b-row>
      <b-col cols="4">
        <b-form-group id="NumSamples-g" label-for="NumSamples-i"
        description="Number of instances of the neighborhood to generate">
          <label for="NumSamples-i">Neighborhood size: {{ neighb_parameters.NumSamples }}</label>
          <b-form-input id="NumSamples-i" v-model="neighb_parameters.NumSamples" required
                        type="number" :min="0" :max="5000" :step="10"/>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="NeighbTypes-g" label-for="NeighbTypes-i"
        description="Select one or more neighborhood generation types">
          <label for="NeighbTypes-i">Neighborhood types:</label>
          <b-form-checkbox
            v-for="option in neighb_types_options"
            v-model="neighb_parameters.NeighbTypes"
            :key="option.value"
            :value="option.value"
            name="flavour-4a"
          >
            {{ option.text }}
          </b-form-checkbox>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row class="mb-4">
      <b-col>
        <b-button @click="sendRequest()" variant="primary">
          <b-spinner small v-if="waitingForResponse">Loading...</b-spinner>
          <span v-else>Submit</span>
        </b-button>
      </b-col>
    </b-row>
  </b-form>
</template>

<style scoped>

</style>
