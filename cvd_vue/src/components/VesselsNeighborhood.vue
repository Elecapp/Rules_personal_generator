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
        <VesselsForm @sendRequest="sendRequest" :waiting-for-response="isLoading"/>
      </b-col>
    </b-row>
    <div id="viz"></div>
  </div>
</template>

<script>
import VesselsForm from './VesselsForm.vue';

export default {
  name: 'VesselsNeighborhood',
  components: {
    VesselsForm,
  },
  data() {
    return {
      isLoading: false,
      contentValue: '',
      spec: {},
    };
  },
  methods: {
    sendRequest(strPayload) {
      const strValue = strPayload;
      this.isLoading = true;

      fetch('/api/vessels/neighborhood/visualization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: strValue,
      })
        .then(response => response.json())
        .then((data) => {
          this.spec = data;
          // Render the visualization
          // embed('#viz', data, { mode: 'vega-lite' });
          vegaEmbed('#viz', data, { mode: 'vega-lite' });
          this.isLoading = false;
        },
        ).catch((error) => {
          alert('There was an error processing your request');
          this.isLoading = false;
        });
    },
  },
};
</script>

<style scoped>

</style>
