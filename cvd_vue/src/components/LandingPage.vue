<script>
export default {
  name: "LandingPage"
}
</script>

<template>
  <div>
    <b-row>
      <b-col>
        <b-card no-body header="Vessels">
          <b-list-group flush>
            <b-list-group-item>
              <b-link :to="{path:'/vessels-neighborhood'}">Neighborhood exploration</b-link>
            </b-list-group-item>
            <b-list-group-item>
              <b-link :to="{path:'/vessels-rules'}">Rules exploration</b-link>
            </b-list-group-item>
          </b-list-group>
        </b-card>
      </b-col>
      <b-col>
        <b-card no-body header="COVID">
          <b-list-group flush>
            <b-list-group-item>
              <b-link :to="{path:'/covid-neighborhood'}">Neighborhood exploration</b-link>
            </b-list-group-item>
            <b-list-group-item>
              <b-link :to="{path:'/covid-rules'}">Rules exploration</b-link>
            </b-list-group-item>
          </b-list-group>
        </b-card>
      </b-col>
    </b-row>
  </div>
</template>

<style scoped>

</style>
