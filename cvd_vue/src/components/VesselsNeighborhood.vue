<template>
  <div>
    <b-row>
      <h1>Vessels Neighborhood Explorer</h1>
      <p></p>
    </b-row>
    <b-row>
      <b-col>
        <p>This interface allows the exploration of a neighborhood generated around a given instance.
        Use the form below to insert the values of the instance to be used as seed for the generation. The
        parameters for the neighborhood generation can be defined at the bottom.</p>
      </b-col>
    </b-row>
    <b-row>
      <b-col>
        <b-form>
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
                  inline
                >
                  {{ option.text }}
                </b-form-checkbox>
              </b-form-group>
            </b-col>
          </b-row>
          <b-row class="mb-4">
            <b-col>
              <b-button @click="sendRequest()" variant="primary">Submit</b-button>
            </b-col>
          </b-row>
        </b-form>
      </b-col>
    </b-row>
    <div id="viz"></div>

  </div>
</template>

<script>
export default {
  name: 'VesselsNeighborhood',
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
      ],
      contentValue: '',
      spec: {},
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
        neighborhood_types: this.neighb_parameters.NeighbTypes,
      };
      console.log('form', request);
      const strValue = JSON.stringify(request);

      fetch('http://localhost:10000/api/vessels/neighborhood/visualization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: strValue,
      })
        .then(response => response.json())
        .then((data) => {
          console.log('data', data);
          this.spec = data;
          // Render the visualization
          // embed('#viz', data, { mode: 'vega-lite' });
          vegaEmbed('#viz', data, {"mode": "vega-lite"});
        },
        ).catch((error) => {
          console.error('Error:', error);
          alert('There was an error processing your request');
        });
    },
  },

};
</script>

<style scoped>

</style>
