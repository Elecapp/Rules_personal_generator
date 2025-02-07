<template>
  <div>
    <b-row>
      <h1>Vessels Neighborhood Explorer</h1>
      <p></p>
    </b-row>
    <b-row>
      <b-form>
        <b-form-group id="SpeedMinimum-g" label-for="SpeedMinimum-i"
        description="The minimum speed in the period">
          <label for="SpeedMinimum-i">Speed: {{form.SpeedMinimum}}</label>
          <b-form-input id="SpeedMinimum-i" v-model="form.SpeedMinimum" required
          type="range" :min="0" :max="20" :step="0.1">

          </b-form-input>
        </b-form-group>
      </b-form>
    </b-row>
    <b-form-textarea id="json-input" placeholder="Insert a JSON object"
                     rows="15"></b-form-textarea>
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
        DistanceStartShapeCurvature: 0,
        DistStartTrendAngle: 0,
        DistStartTrendDevAmplitude: 0,
        MaxDistPort: 0,
        MinDistPort: 0,
      },
    };
  },

};
</script>

<style scoped>

</style>
